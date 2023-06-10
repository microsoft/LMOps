import random
from datasets import load_dataset
import re
import tqdm
import glob
import logging

logger = logging.getLogger(__name__)


train_cluster_map = {
    "close_qa": ["natural_questions", "arc_c", "arc_e"],
    "common_reason": ["copa", "piqa", "hellaswag"],
    "coreference": ["winogrande", "wsc"],
    "nli": ["mnli", "qnli", "rte", "snli"],
    "paraphrase": ["mrpc", "paws", "qqp"],
    "reading": ["multirc", "openbookqa", "squad_v1", "boolq"],
    "sentiment": ["yelp", "sentiment140", "sst2"],
    "struct2text": ["common_gen", "e2e_nlg", "dart"],
    "summarize": ["aeslc", "ag_news", "gigaword"],

    # train examples for a quick try
    "train_example_1": ["rte"],
    "train_example_2": ["copa", "piqa"],

    # cot prompting example
    "cot_train_example": ["pubmed_qa"]
}
test_cluster_map = {
    "close_qa": ["natural_questions", "arc_c", "arc_e"],
    "common_reason": ["copa", "piqa", "hellaswag"],
    "coreference": ["winogrande", "wsc", "wsc273"],
    "nli": ["rte", "mnli_m", "mnli_mm", "qnli", "snli"],
    "paraphrase": ["mrpc", "paws", "qqp"],
    "reading": ["multirc", "openbookqa", "squad_v1", "boolq"],
    "sentiment": ["yelp", "sentiment140", "sst2"],
    "struct2text": ["common_gen", "e2e_nlg", "dart"],
    "summarize": ["aeslc", "ag_news", "gigaword"],
    
    # test examples for a quick try
    "test_example_1": ["arc_e"],
    "test_example_2": ["mrpc"],
    
    # cot prompting example
    "cot_test_example": ["pubmed_qa"]
}


def get_prompt_files(prompt_pool_path, train_clusters: str):
    """
    train_clusters: a string representing all clusters concatenated by `+`,
    e.g.,
    `nli+close_qa` denotes nli and close_qa clusters
    """
    clusters=train_clusters.split('+')
    prompt_pool_dirs = [
        f"{prompt_pool_path}/{cluster}/*"
        for cluster in clusters
    ]
    logger.info("prompt pool dirs: %s", prompt_pool_dirs)
    prompt_pool_paths = []
    for dir in prompt_pool_dirs:
        prompt_pool_paths.extend(glob.glob(dir))
    return prompt_pool_paths


class App:
    def __init__(self):
        self.cls_dic = {}

    def add(self, key):
        def adder(cls):
            self.cls_dic[key] = cls
            return cls
        return adder


class BaseTask(object):
    def __init__(self):
        self.finder_L = 50  # num of prompts to be sampled from the pool for scoring
        self.run_scorer_bsz = 5 # batch size per GPU for scoring prompts
        self.balance_class = False # whether to balance data examples sampled from each class
    
    def filter(self, entry): # data filter
        return True

    def load_data_split(self, dataset, ds_size=None, split="train"):
        assert split in ["train", 'validation', 'test']
        if split in ["test", "validation"]:
            assert ds_size == None, "should not split test/valid set"
        if ds_size == None or ds_size == "None":
            data_split = list(dataset[split])
            return [entry for entry in data_split if self.filter(entry)]
        data = dataset[split]
        data = list(data.shuffle(seed=42))
        data = [entry for entry in data if self.filter(entry)]
        if self.balance_class:  # balance data examples sampled from each class
            counts = [0] * self.class_num
            num_each_class = ds_size // self.class_num
            x = []
            for entry in tqdm.tqdm(data):
                if len(x) >= ds_size:
                    break
                label = self.get_label(entry)
                if counts[label] >= num_each_class:
                    continue
                counts[label] += 1
                x.append(entry)
        else:
            x = data[:ds_size]
        return x

    def get_template(self, entry, return_answer = False):
        '''
        random sample a template for each entry
        '''
        if return_answer:
            templates = [p[1] for p in self.get_templates()]
        else:
            templates = [p[0] for p in self.get_templates()]
        random.seed(entry["id"]) # fix random seed for reproduction
        template = random.choice(templates)
        return template


task_map = App()

# ==================Natural Language Inference========================
@task_map.add("mnli")
class Mnli(BaseTask):
    def __init__(self):
        super().__init__()
        self.class_num = 3
        self.metric = "simple_accuracy"
        self.balance_class = True
        self.cluster = "nli"

    def get_dataset(self, split=None, ds_size=None, cache_dir=None):
        dataset = load_dataset("glue", "mnli", cache_dir=cache_dir)
        if split == "train":
            return self.load_data_split(dataset, ds_size=ds_size, split=split)
        else:  
            raise Exception(
                "Please switch to mnli_matched/mis_matched for mnli validation sets"
            )
        
    def get_templates(self):
        return [
                ("Premise: \"{premise}\" Hypothesis: \"{hypothesis}\" Does the premise entail the hypothesis? Yes, No, or Maybe?", "{answer}"),
                ("Premise: \"{premise}\" Hypothesis: \"{hypothesis}\" Is the hypothesis entailed by the premise? Yes, No, or Maybe?", "{answer}"),
                ("Here is a premise: \"{premise}\" Here is a hypothesis: \"{hypothesis}\" Is it possible to conclude that if the premise is true, then so is the hypothesis? Yes, No, or Maybe?", "{answer}"),
                ("Sentence 1: \"{premise}\" Sentence 2: \"{hypothesis}\" Is this second sentence entailed by the first sentence? Yes, No, or Maybe?", "{answer}"),
                ("Sentence 1: \"{premise}\" Sentence 2: \"{hypothesis}\" If the first sentence is true, then is the second sentence true? Yes, No, or Maybe?", "{answer}"),
                ("Based on the premise \"{premise}\", can we conclude the hypothesis \"{hypothesis}\" is true? Yes, No, or Maybe?", "{answer}"),
                ("Premise: \"{premise}\" If this premise is true, what does that tell us about whether it entails the hypothesis \"{hypothesis}\"? Yes, No, or Maybe?", "{answer}"),
                ("Premise: \"{premise}\" Based on this premise, is the hypothesis \"{hypothesis}\" true? Yes, No, or Maybe?", "{answer}"),
                ("If \"{premise}\", can we conclude that \"{hypothesis}\"? Yes, No, or Maybe?", "{answer}"),
                ("\"{premise}\" Does it follow that \"{hypothesis}\"? Yes, No, or Maybe?", "{answer}"),
            ]
 
    def get_question(self, entry):
        premise = entry["premise"]
        hypothesis = entry["hypothesis"]
        template = self.get_template(entry)
        return template.replace("{premise}", premise).replace("{hypothesis}", hypothesis)

    def get_input_strs(self, entry):
        text = self.get_question(entry)
        return [text] * self.class_num

    def get_answers(self, entry):
        answers = [" Yes", " Maybe", " No"]
        return answers

    def get_label(self, entry):
        label = int(entry["label"])
        return label

    def get_answer(self, entry):
        answers = self.get_answers(entry)
        label = self.get_label(entry)
        return answers[label]


@task_map.add("mnli_m")
class Mnli_m(Mnli):
    def __init__(self):
        super().__init__()

    def get_dataset(self, split=None, ds_size=None, cache_dir=None):
        dataset = load_dataset("glue", "mnli_matched", cache_dir=cache_dir)
        if split == "train":
            raise Exception("Please switch to mnli for mnli training sets")
        else:  
            split = "validation"
            return self.load_data_split(dataset, split=split)


@task_map.add("mnli_mm")
class Mnli_mm(Mnli):
    def __init__(self):
        super().__init__()

    def get_dataset(self, split=None, ds_size=None, cache_dir=None):
        dataset = load_dataset("glue", "mnli_mismatched", cache_dir=cache_dir)
        if split == "train":
            raise Exception("Please switch to mnli for mnli training sets")
        else:  
            split = "validation"
            return self.load_data_split(dataset, split=split)


@task_map.add("qnli")
class Qnli(BaseTask):
    def __init__(self):
        super().__init__()
        self.class_num = 2
        self.metric = "simple_accuracy"
        self.balance_class = True
        self.cluster = "nli"

    def get_dataset(self, split=None, ds_size=None, cache_dir=None):
        dataset = load_dataset("glue", "qnli", cache_dir=cache_dir)
        if split == "train":
            return self.load_data_split(dataset, ds_size=ds_size, split=split)
        else:  
            split = "validation"
            return self.load_data_split(dataset, split=split)

    def get_templates(self):
        return [
                ("Does the sentence \"{sentence}\" answer the question \"{question}\"?", "{answer}"),
                ("Does the sentence \"{sentence}\" provide a valid answer to the question \"{question}\"?", "{answer}"),
                ("Is \"{sentence}\" a good answer to the question \"{question}\"?", "{answer}"),
                ("Does \"{sentence}\" correctly answer the question of \"{question}\"?", "{answer}"),
                ("Does \"{sentence}\" contain the correct answer to \"{question}\"?", "{answer}"),
                ("Q: {question}  A: {sentence}  Does the answer correctly answer the question?", "{answer}"),
                ("Question: {question} Answer: {sentence}  Is the question answered in a satisfactory fashion?", "{answer}"),
                ("Question: {question} Is {sentence} a good answer to this question?", "{answer}"),
                ("Question: {question} Is \"{sentence}\" the correct answer?", "{answer}"),
            ]

    def get_question(self, entry):
        question = entry["question"]
        sentence = entry["sentence"]
        template = self.get_template(entry)
        return template.replace("{question}", question).replace("{sentence}", sentence)

    def get_input_strs(self, entry):
        text = self.get_question(entry)
        return [text] * self.class_num

    def get_answers(self, entry):
        answers = [" Yes", " No"]
        return answers

    def get_label(self, entry):
        label = int(entry["label"])
        return label

    def get_answer(self, entry):
        answers = self.get_answers(entry)
        label = self.get_label(entry)
        return answers[label]


@task_map.add("rte")
class Rte(BaseTask):
    def __init__(self):
        super().__init__()
        self.class_num = 3
        self.metric = "simple_accuracy"
        self.balance_class = True
        self.cluster = "nli"

    def get_dataset(self, split=None, ds_size=None, cache_dir=None):
        dataset = load_dataset("super_glue", "rte", cache_dir=cache_dir)
        if split == "train":
            return self.load_data_split(dataset, ds_size=ds_size, split=split)
        else:  
            split = "validation"
            return self.load_data_split(dataset, split=split)

    def get_templates(self):
        return [
                ("{premise} Based on the paragraph above can we conclude that \"{hypothesis}\"? Yes, No, or Maybe?", "{answer}"),
                ("{premise} Based on that paragraph can we conclude that this sentence is true? {hypothesis} Yes, No, or Maybe?", "{answer}"),
                ("{premise} Can we draw the following conclusion? {hypothesis} Yes, No, or Maybe?", "{answer}"),
                ("{premise} Does this next sentence follow, given the preceding text? {hypothesis} Yes, No, or Maybe?", "{answer}"),
                ("{premise} Can we infer the following? {hypothesis} Yes, No, or Maybe?", "{answer}"),
                ("Read the following paragraph and determine if the hypothesis is true: {premise} Hypothesis: {hypothesis} Yes, No, or Maybe?", "{answer}"),
                ("Read the text and determine if the sentence is true: {premise} Sentence: {hypothesis} Yes, No, or Maybe?", "{answer}"),
                ("Can we draw the following hypothesis from the context?  Context: {premise} Hypothesis: {hypothesis} Yes, No, or Maybe?", "{answer}"),
                ("Determine if the sentence is true based on the text below: {hypothesis} {premise} Yes, No, or Maybe?", "{answer}"),
            ]

    def get_question(self, entry):
        premise = entry["premise"]
        hypothesis = entry["hypothesis"]
        template = self.get_template(entry)
        return template.replace("{premise}", premise).replace(
            "{hypothesis}", hypothesis
        )

    def get_input_strs(self, entry):
        text = self.get_question(entry)
        return [text] * self.class_num

    def get_answers(self, entry):
        answers = [" Yes", " Maybe", " No"]
        return answers

    def get_label(self, entry):
        label = int(entry["label"])
        return label

    def get_answer(self, entry):
        answers = self.get_answers(entry)
        label = self.get_label(entry)
        return answers[label]

# define your multiple-choice task
@task_map.add("snli")
class Snli(BaseTask):
    def __init__(self):
        super().__init__()
        self.class_num = 3 # number of class
        self.metric = "simple_accuracy" # metric name, should be the same as in `./src/utils/metric.py``
        self.balance_class = True # whether to balance number of data examples sampled from each class
        self.cluster = "nli" # task cluster name, should be the same as in the task map at the top of this file

    # filter to remove some unexpected data samples, 
    # if nothing to be removed, just delete this function
    def filter(self, entry):
        return int(entry["label"]) >= 0

    # get dataset splits
    def get_dataset(self, split=None, ds_size=None, cache_dir=None):
        dataset = load_dataset("snli", cache_dir=cache_dir) # automatically download datasets into the `cache_dir`
        if split == "train":
            return self.load_data_split(dataset, ds_size=ds_size, split=split)
        else:  
            # load validation/test split for evaluation
            split = "test"
            return self.load_data_split(dataset, split=split)

    # define task templates to transfer the datasets to instructions, 
    # we use the same templates with FLAN: https://github.com/google-research/FLAN/blob/main/flan/templates.py
    # Remove the newline character and option suffices for better prompting performance.
    def get_templates(self):
        return [
                ("If \"{premise}\", does this mean that \"{hypothesis}\"? Yes, No, or Maybe?", "{answer}"),
                ("If \"{premise}\", can we conclude \"{hypothesis}\"? Yes, No, or Maybe?", "{answer}"),
                ("If \"{premise}\", does it logically follow that \"{hypothesis}\"? Yes, No, or Maybe?", "{answer}"),
                ("Based on the sentence \"{premise}\", is the sentence \"{hypothesis}\" a true sentence? Yes, No, or Maybe?", "{answer}"),
                ("Premise: {premise} Hypothesis: {hypothesis} Can we conclude that the hypothesis is true if the premise is true? Yes, No, or Maybe?", "{answer}"),
                ("Premise: {premise} Hypothesis: {hypothesis} Given the premise, can we conclude the hypothesis? Yes, No, or Maybe?", "{answer}"),
                ("Here is a premise: \"{premise}\" Here is a hypothesis: \"{hypothesis}\". Does the premise tell us whether the hypothesis is true? Yes, No, or Maybe?", "{answer}"),
                ("Is it possible to conclude that \"{premise}\" if \"{hypothesis}\"? Yes, No, or Maybe?", "{answer}"),
                ("Is the premise \"{premise}\" true if \"{hypothesis}\"? Yes, No, or Maybe?", "{answer}"),
            ]

    # random_sample one template to convert the task input to an instruction
    def get_question(self, entry): 
        premise = entry["premise"]
        hypothesis = entry["hypothesis"]
        template = self.get_template(entry)
        return template.replace("{premise}", premise).replace("{hypothesis}", hypothesis)

    # wrap questions as a list for scoring/inference
    def get_input_strs(self, entry):
        text = self.get_question(entry)
        return [text] * self.class_num

    # get all candidate options
    def get_answers(self, entry):
        answers = [" Yes", " Maybe", " No"]
        return answers

    # get index of the gold option
    def get_label(self, entry):
        label = int(entry["label"])
        return label

    # get gold option string
    def get_answer(self, entry): 
        answers = self.get_answers(entry)
        label = self.get_label(entry)
        return answers[label]


# ==============================Reading Comprehension================================
@task_map.add("boolq")
class Boolq(BaseTask):
    def __init__(self):
        super().__init__()
        self.class_num = 2
        self.metric = "simple_accuracy"
        self.balance_class = True
        self.cluster = "reading"

    def get_dataset(self, split=None, ds_size=None, cache_dir=None):
        dataset = load_dataset("super_glue", "boolq", cache_dir=cache_dir)
        if split == "train":
            return self.load_data_split(dataset, ds_size=ds_size, split=split)
        else:  
            split = "validation"
            return self.load_data_split(dataset, split=split)

    def get_templates(self):
        return [
                ("{text} Can we conclude that {question}?", "{answer}"),
                ("{text} Is it true that {question}?", "{answer}"),
                ("{text} {question}?", "{answer}"),
                ("Text: {text} Question: {question}?", "{answer}"),
                ("{text} What's the best answer to this question: {question}?", "{answer}"),
                ("{text} Based on the above text, what's the best answer to this question: {question}?", "{answer}"),
                ("{text} Answer this question, making sure that the answer is supposed by the text: {question}?", "{answer}"),
                ("{text} Is the following statement correct based on the text {question}", "{answer}"),
                ("{text} Is this statement correct \"{question}\"?", "{answer}"),
                ("Is it true that {question} based on the following text? {text}", "{answer}"),
            ]

    def get_question(self, entry):
        text = entry["passage"]
        question = entry["question"]
        template = self.get_template(entry)
        return template.replace("{text}", text).replace("{question}", question)

    def get_input_strs(self, entry):
        text = self.get_question(entry)
        return [text] * self.class_num

    def get_answers(self, entry):
        answers = [" No", " Yes"]
        return answers

    def get_label(self, entry):
        label = int(entry["label"])
        return label

    def get_answer(self, entry):
        answers = self.get_answers(entry)
        label = self.get_label(entry)
        return answers[label]

@task_map.add("multirc")
class Multirc(BaseTask):
    def __init__(self):
        super().__init__()
        self.class_num = 2
        self.metric = "f1"
        self.balance_class = True
        self.cluster = "reading"
        self.run_scorer_bsz = 1 # set the bsz a smaller value when the input text is too long

    def get_dataset(self, split=None, ds_size=None, cache_dir=None):
        dataset = load_dataset("super_glue", "multirc", cache_dir=cache_dir)
        if split == "train":
            return self.load_data_split(dataset, ds_size=ds_size, split=split)
        else:  
            split = "validation"
            return self.load_data_split(dataset, split=split)

    def get_templates(self):
        return [
                ("{paragraph} Question: \"{question}\" Response: \"{response}\" Does the response correctly answer the question?", "{answer}"),
                ("{paragraph} Question: \"{question}\" Response: \"{response}\" Based on the paragraph, is the response to the question is factually correct?", "{answer}"),
                ("{paragraph} Question: \"{question}\" Answer: \"{response}\" Is this answer correct?", "{answer}"),
                ("Paragraph: {paragraph} Question: \"{question}\" Answer: \"{response}\" Based on the paragraph, is this answer correct", "{answer}"),
                ("{paragraph} Based on the paragraph, does the response \"{response}\" correctly answer the question \"{question}\"?", "{answer}"),
                ("{paragraph} According to the above paragraph, the correct answer to the question \"{question}\" is \"{response}\"?", "{answer}"),
                ("{paragraph} After reading the above, is \"{response}\" the correct answer to the question \"{question}\"?", "{answer}"),
                ("{paragraph} Question: \"{question}\" Answer: \"{response}\" Is this answer to the question correct?", "{answer}"),
            ]

    def get_question(self, entry):
        paragraph = entry["paragraph"]
        question = entry["question"]
        response = entry["answer"]
        template = self.get_template(entry)
        return template.replace("{paragraph}", paragraph).replace("{question}", question).replace("{response}", response)

    def get_input_strs(self, entry):
        text = self.get_question(entry)
        return [text] * self.class_num

    def get_answers(self, entry):
        answers = [" No", " Yes"]
        return answers

    def get_label(self, entry):
        label = int(entry["label"])
        return label

    def get_answer(self, entry):
        answers = self.get_answers(entry)
        label = self.get_label(entry)
        return answers[label]


@task_map.add("openbookqa")
class Openbookqa(BaseTask):
    def __init__(self):
        super().__init__()
        self.class_num = 4
        self.metric = "simple_accuracy"
        self.balance_class = False
        self.cluster = "reading"
        self.finder_L = 200

    def get_dataset(self, split=None, ds_size=None, cache_dir=None):
        dataset = load_dataset("openbookqa", "additional", cache_dir=cache_dir)
        if split == "train":
            return self.load_data_split(dataset, ds_size=ds_size, split=split)
        else:  
            split = "test"
            return self.load_data_split(dataset, split=split)

    def get_templates(self):
        return [
                ("{fact} {question}", "{answer}"),
                ("Read this fact: \"{fact}\" Now answer this question: \"{question}\"", "{answer}"),
                ("Given the fact \"{fact}\", what is the answer to the question or completion \"{question}\"", "{answer}"),
                ("Knowing that \"{fact}\", how would one answer \"{question}\"", "{answer}"),
                ("Use evidence from the fact that {fact} to answer this question: \"{question}\"", "{answer}"),
                ("Fact: {fact} Question: {question} What's the answer?", "{answer}"),
                ("Use this fact to answer the question: {fact} {question}", "{answer}"),
            ]

    def get_question(self, entry):
        fact = entry["fact1"]
        question = entry["question_stem"]
        template = self.get_template(entry)
        return template.replace("{fact}", fact).replace("{question}", question)

    def get_input_strs(self, entry):
        text = self.get_question(entry)
        return [text] * self.class_num

    def get_answers(self, entry):
        answers = [' '+answer for answer in entry["choices"]["text"]]
        return answers

    def get_label(self, entry):
        answerKey = entry["answerKey"]
        label = {"A": 0, "B": 1, "C": 2, "D": 3}[answerKey]
        return label

    def get_answer(self, entry):
        answers = self.get_answers(entry)
        label = self.get_label(entry)
        return answers[label]


@task_map.add("squad_v1")
class Squad_v1(BaseTask):
    def __init__(self):
        super().__init__()
        self.class_num = 1 # the number of options of text completion is always set as 1.
        self.metric = "squad"
        self.balance_class = False
        self.cluster = "reading"

        # set the number of prompts a larger value for difficult question, 
        # ensure we could find a least one prompt scoring than 0 in the sampled prompts
        self.finder_L = 100

    def get_dataset(self, split=None, ds_size=None, cache_dir=None):
        dataset = load_dataset("squad", cache_dir=cache_dir)
        if split == "train":
            return self.load_data_split(dataset, ds_size=ds_size, split=split)
        else:  
            split = "validation"
            return self.load_data_split(dataset, split=split)

    def get_templates(self):
        return [
                ("Please answer a question about the following article about {title}: {context} {question}", "{answer}"),
                ("Read this and answer the question {context} {question}", "{answer}"),
                ("{context} {question}", "{answer}"),
                ("Answer a question about this article: {context} {question}", "{answer}"),
                ("Here is a question about this article: {context} What is the answer to this question: {question}", "{answer}"),
                ("Article: {context} Question: {question}", "{answer}"),
                ("Article: {context} Now answer this question: {question}", "{answer}"),
                ("{title} {context} Q: {question}", "{answer}"),
            ]

    def get_question(self, entry):
        title = re.sub(r"_", " ", entry["title"])
        context = entry["context"]
        question = entry["question"]
        template = self.get_template(entry)
        return (
            template.replace("{context}", context)
            .replace("{question}", question)
            .replace("{title}", title)
        )

    # wrap the input str as a list to align with multiple choice task
    def get_input_strs(self, entry):
        text = self.get_question(entry)
        return [text]

    # wrap the gold answer as a list to align with multiple choice task
    def get_answers(self, entry):
        answers = [' '+entry["answers"]["text"][0]]
        return answers

    # get label completion(s), the squad metric requires the label to be a list of string(s)
    # get_label is for caculating metric scores, so we need to return all label strings
    def get_label(self, entry):
        label = entry["answers"]["text"]
        return label

    # get_answer function is for constructing demonstration in the prompt pool,
    # return a string
    def get_answer(self, entry):
        return ' '+entry["answers"]["text"][0]


# ===================================Commonsense Reasoning===============================
@task_map.add("copa")
class Copa(BaseTask):
    def __init__(self):
        super().__init__()
        self.class_num = 2
        self.metric = "simple_accuracy"
        self.balance_class = False
        self.cluster = "common_reason"
        self.finder_L = 200

    def get_dataset(self, split=None, ds_size=None, cache_dir=None):
        dataset = load_dataset("super_glue", "copa", cache_dir=cache_dir)
        if split == "train":
            return self.load_data_split(dataset, ds_size=ds_size, split=split)
        else:  
            split = "validation"
            return self.load_data_split(dataset, split=split)

    def get_templates(self):
        return [
                ("\"{premise}\" What is the {question}?", "{answer}"),
                ("Here is a premise: \"{premise}\" What is the {question}?", "{answer}"),
                ("\"{premise}\" What is the {question} of the preceding sentence?", "{answer}"),
                ("\"{premise}\" What is a plausible {question}?", "{answer}"),
                ("Based on the following sentence, what is the {question}? \"{premise}\"", "{answer}"),
                ("\"{premise}\" {question}: ", "{answer}"),
                ("What is the {question} of the following sentence? \"{premise}\"", "{answer}"),
                ("Answer the following question about this sentence: \"{premise}\" What is the {question}?", "{answer}"),
             ]

    def get_question(self, entry):
        question = entry["question"]
        premise = entry["premise"]
        template = self.get_template(entry)
        return template.replace("{question}", question).replace("{premise}", premise)

    def get_input_strs(self, entry):
        text = self.get_question(entry)
        return [text] * self.class_num

    def get_answers(self, entry):
        answers = [' '+entry["choice1"], ' '+entry["choice2"]]
        return answers

    def get_label(self, entry):
        label = int(entry["label"])
        return label

    def get_answer(self, entry):
        answers = self.get_answers(entry)
        label = self.get_label(entry)
        return answers[label]

@task_map.add("hellaswag")
class Hellaswag(BaseTask):
    def __init__(self):
        super().__init__()
        self.class_num = 4
        self.metric = "simple_accuracy"
        self.balance_class = False
        self.cluster = "common_reason"

    def get_dataset(self, split=None, ds_size=None, cache_dir=None):
        dataset = load_dataset("hellaswag", cache_dir=cache_dir)
        if split == "train":
            return self.load_data_split(dataset, ds_size=ds_size, split=split)
        else:  
            split = "validation"
            return self.load_data_split(dataset, split=split)

    def get_templates(self):
        return [
                ("What happens next in this paragraph? {context}", "{answer}"),
                ("Continue writing the next sentence in this paragraph: {context}", "{answer}"),
                ("Continue writing the next sentence. {context}", "{answer}"),
                ("This is a test of commonsense. Complete the next sentence: {context}", "{answer}"),
                ("Write the next sentence in this paragraph: {context}", "{answer}"),
                ("How does the next paragraph end? {context}", "{answer}"),
                ("What most naturally follows? {context}", "{answer}"),
                ("What happens next? {context}", "{answer}"),
                ("What is the most logical next event? {context}", "{answer}"),
                ("Write the next sentence in the following story. {context}", "{answer}"),
            ]

    def get_question(self, entry):
        # Model will likely have a hard time producing a string with brackets.
        # context=re.sub(r'\[header\]\s', '', entry['ctx'])
        context = re.sub(r"\[.*?\]\s", "", entry["ctx"])
        template = self.get_template(entry)
        return template.replace("{context}", context)

    def get_input_strs(self, entry):
        text = self.get_question(entry)
        return [text] * self.class_num

    def get_answers(self, entry):
        answers = []
        for answer in entry["endings"]:
            answers.append(re.sub(r"\[.*?\]\s", "", answer))
        return [' '+ answer for answer in answers]

    def get_label(self, entry):
        label = int(entry["label"])
        return label

    def get_answer(self, entry):
        answers = self.get_answers(entry)
        label = self.get_label(entry)
        return answers[label]


@task_map.add("piqa")
class Piqa(BaseTask):
    def __init__(self):
        super().__init__()
        self.class_num = 2
        self.metric = "simple_accuracy"
        self.balance_class = False
        self.cluster = "common_reason"

    def get_dataset(self, split=None, ds_size=None, cache_dir=None):
        dataset = load_dataset("piqa", cache_dir=cache_dir)
        if split == "train":
            return self.load_data_split(dataset, ds_size=ds_size, split=split)
        else:  
            split = "validation"
            return self.load_data_split(dataset, split=split)

    def get_templates(self):
        return [
                ("Here is a goal: \"{goal}\" How would you accomplish this goal?", "{answer}"),
                ("Here is a goal: \"{goal}\" Which way makes more sense to accomplish this goal?", "{answer}"),
                ("Goal: \"{goal}\" Which of the following methods is more reasonable for accomplishing this goal?", "{answer}"),
                ("BaseTaskive: \"{goal}\" Which of the following solutions is more sound in terms of naive physics reasoning?", "{answer}"),
                ("How do you do this: \"{goal}\"", "{answer}"),
                ("What is the best way to: \"{goal}\"", "{answer}"),
                ("Which of the following solutions is better for the following goal: \"{goal}\"", "{answer}"),
                ("How would someone go about accomplishing this goal? \"{goal}\"", "{answer}"),
            ]

    def get_question(self, entry):
        goal = entry["goal"]
        template = self.get_template(entry)
        return template.replace("{goal}", goal)

    def get_input_strs(self, entry):
        text = self.get_question(entry)
        return [text] * self.class_num

    def get_answers(self, entry):
        answers = [' '+entry["sol1"], ' '+entry["sol2"]]
        return answers

    def get_label(self, entry):
        label = int(entry["label"])
        return label

    def get_answer(self, entry):
        answers = self.get_answers(entry)
        label = self.get_label(entry)
        return answers[label]


# ================================Sentiment Analysis=============================
@task_map.add("sentiment140")
class Sentiment140(BaseTask):
    def __init__(self):
        super().__init__()
        self.class_num = 2
        # Prior work uses two classes
        # (https://www.aclweb.org/anthology/C14-1008.pdf,
        # https://arxiv.org/pdf/1404.2188.pdf)
        self.metric = "simple_accuracy"
        self.balance_class = True
        self.cluster = "sentiment"

    def filter(self, entry):
        return int(entry["sentiment"]) in [0, 4] # Prior work uses two classes

    def get_dataset(self, split=None, ds_size=None, cache_dir=None):
        dataset = load_dataset("sentiment140", cache_dir=cache_dir)
        if split == "train":
            return self.load_data_split(dataset, ds_size=ds_size, split=split)
        else:  
            split = "test"
            return self.load_data_split(dataset, split=split)

    def get_templates(self):
        return [
            ("{text} What is the sentiment of this tweet?", "{answer}"),
            ("{text} How would the sentiment of this tweet be described?", "{answer}"),
            ("{text} Describe the sentiment embodied by this tweet.", "{answer}"),
            ("Tweet: {text} Predict the sentiment of this tweet.", "{answer}"),
            ("What is the sentiment of the following tweet? Tweet:{text}", "{answer}"),
            ("How would one describe the sentiment of this tweet? {text}", "{answer}"),
        ]

    def get_question(self, entry):
        text = entry["text"]
        template = self.get_template(entry)
        return template.replace("{text}", text)

    def get_input_strs(self, entry):
        text = self.get_question(entry)
        return [text] * self.class_num

    def get_answers(self, entry):
        answers = [" Negative", " Positive"]
        return answers

    def get_label(self, entry):
        label = 0 if int(entry["sentiment"]) == 0 else 1
        return label

    def get_answer(self, entry):
        answers = self.get_answers(entry)
        label = self.get_label(entry)
        return answers[label]


@task_map.add("sst2")
class Sst2(BaseTask):
    def __init__(self):
        super().__init__()
        self.class_num = 2
        self.metric = "simple_accuracy"
        self.balance_class = True
        self.cluster = "sentiment"

    def get_dataset(self, split=None, ds_size=None, cache_dir=None):
        dataset = load_dataset("sst2", cache_dir=cache_dir)
        if split == "train":
            return self.load_data_split(dataset, ds_size=ds_size, split=split)
        else:  
            split = "validation"
            return self.load_data_split(dataset, split=split)

    def get_templates(self):
        return [
                ("Review: \"{sentence}\" Is this movie review sentence negative or positive?", "{answer}"),
                ("Short movie review: \"{sentence}\" Did the critic thinking positively or negatively of the movie?", "{answer}"),
                ("Sentence from a movie review: \"{sentence}\" Was the movie seen positively or negatively based on the preceding review?", "{answer}"),
                ("\"{sentence}\" How would the sentiment of this sentence be perceived?", "{answer}"),
                ("Is the sentiment of the following sentence positive or negative? \"{sentence}\"", "{answer}"),
                ("What is the sentiment of the following movie review sentence? \"{sentence}\"", "{answer}"),
                ("Would the following phrase be considered positive or negative? \"{sentence}\"", "{answer}"),
                ("Does the following review have a positive or negative opinion of the movie? \"{sentence}\"", "{answer}"),
            ]

    def get_question(self, entry):
        sentence = entry["sentence"]
        template = self.get_template(entry)
        return template.replace("{sentence}", sentence)

    def get_input_strs(self, entry):
        text = self.get_question(entry)
        return [text] * self.class_num

    def get_answers(self, entry):
        answers = [" Negative", " Positive"]
        return answers

    def get_label(self, entry):
        label = int(entry["label"])
        return label

    def get_answer(self, entry):
        answers = self.get_answers(entry)
        label = self.get_label(entry)
        return answers[label]


@task_map.add("yelp")
class Yelp(BaseTask):
    def __init__(self):
        super().__init__()
        self.class_num = 2
        self.metric = "simple_accuracy"
        self.balance_class = True
        self.cluster = "sentiment"

    def filter(self, entry):
        text = entry["text"]
        text = re.sub(r"\\\"", "", text)
        text = re.sub(r"\\n\\n", " ", text)
        # filter out texts longer than 256
        return len(text) > 0 and len(text.split(" ")) <= 256

    def get_dataset(self, split=None, ds_size=None, cache_dir=None):
        dataset = load_dataset("yelp_polarity", cache_dir=cache_dir)
        if split == "train":
            return self.load_data_split(dataset, ds_size=ds_size, split=split)
        else:  
            split = "test"
            return self.load_data_split(dataset, split=split)

    def get_templates(self):
        return [
                ("{text} Is this review positive or negative?", "{answer}"),
                ("{text} What is the sentiment of this review?", "{answer}"),
                ("{text} Was this review given positively or negatively?", "{answer}"),
                ("{text} How would this review be described in terms of sentiment?", "{answer}"),
                ("Is the following review positive or negative? {text}", "{answer}"),
                ("What is the sentiment of the following review? {text}", "{answer}"),
                ("How might one describe the sentiment of this review? {text}", "{answer}"),
            ]

    def get_question(self, entry):
        text = entry["text"]
        text = re.sub(r"\\\"", "", text)
        text = re.sub(r"\\n\\n", " ", text)
        template = self.get_template(entry)
        return template.replace("{text}", text)

    def get_input_strs(self, entry):
        text = self.get_question(entry)
        return [text] * self.class_num

    def get_answers(self, entry):
        answers = [" Negative", " Positive"]
        return answers

    def get_label(self, entry):
        label = int(entry["label"])
        return label

    def get_answer(self, entry):
        answers = self.get_answers(entry)
        label = self.get_label(entry)
        return answers[label]


# =============================closedbook_qa====================================
@task_map.add("arc_c")
class Arc_c(BaseTask):
    def __init__(self):
        super().__init__()
        self.class_num = 4
        self.metric = "simple_accuracy"
        self.balance_class = False
        self.finder_L = 400
        self.cluster = "close_qa"

    def filter(self, entry):
        return len(entry["choices"]["text"]) == 4

    def get_dataset(self, split=None, ds_size=None, cache_dir=None):
        dataset = load_dataset("ai2_arc", "ARC-Challenge", cache_dir=cache_dir)
        if split == "train":
            return self.load_data_split(dataset, ds_size=ds_size, split=split)
        else:  
            split = "test"
            return self.load_data_split(dataset, split=split)

    def get_templates(self):
        return [
                ("{question}", "{answer}"),
                ("Question: {question} Answer:", "{answer}"),
                ("Question: {question} What is the correct answer to the question from the following choices?", "{answer}"),
                ("Q: {question} What is the correct answer to this question?", "{answer}"),
                ("What is the answer? {question}", "{answer}"),
                ("Answer the question {question}", "{answer}"),
                ("{question} Pick the answer from these options.", "{answer}"),
             ]

    def get_question(self, entry):
        question = entry["question"]
        template = self.get_template(entry)
        return template.replace("{question}", question)

    def get_input_strs(self, entry):
        text = self.get_question(entry)
        assert len(entry["choices"]["text"]) == 4
        return [text] * self.class_num

    def get_answers(self, entry):
        answers = [" " + answer for answer in entry["choices"]["text"]]
        return answers

    def get_label(self, entry):
        # NOTE: Some `entry["answerKey"]`s are in numeric string format being one
        # of {'1', '2', '3', '4'}. We map them back to letters.
        num_to_letter = {"1": "A", "2": "B", "3": "C", "4": "D"}
        entry["answerKey"] = num_to_letter.get(entry["answerKey"], entry["answerKey"])
        label = ["A", "B", "C", "D"].index(entry["answerKey"])
        return label

    def get_answer(self, entry):
        answers = self.get_answers(entry)
        label = self.get_label(entry)
        return answers[label]


@task_map.add("arc_e")
class Arc_e(Arc_c):
    def __init__(self):
        super().__init__()

    def get_dataset(self, split=None, ds_size=None, cache_dir=None):
        dataset = load_dataset("ai2_arc", "ARC-Easy", cache_dir=cache_dir)
        if split == "train":
            return self.load_data_split(dataset, ds_size=ds_size, split=split)
        else:  
            split = "test"
            return self.load_data_split(dataset, split=split)


@task_map.add("natural_questions")
class Natural_questions(BaseTask):
    def __init__(self):
        super().__init__()
        self.class_num = 1
        self.metric = "trivia_qa"
        self.balance_class = False
        self.cluster = "close_qa"
        self.run_scorer_bsz = 20
        self.finder_L = 400

    def get_dataset(self, split=None, ds_size=None, cache_dir=None):
        dataset = load_dataset("nq_open", cache_dir=cache_dir)
        if split == "train":
            return self.load_data_split(dataset, ds_size=ds_size, split=split)
        else:  
            split = "validation"
            return self.load_data_split(dataset, split=split)

    def get_templates(self):
        return [
            ("Question: {question} Answer:", "{answer}"),
            ("{question}", "{answer}"),
            ("Answer the following question: {question}", "{answer}"),
            ("Answer this question: {question}", "{answer}"),
            ("Please answer this question: {question}", "{answer}"),
            ("Answer the question...{question}", "{answer}"),
            ("What is the answer to this question? {question}", "{answer}"),
            ("Can you tell me the answer to {question}", "{answer}"),
            ("Next question: {question}", "{answer}"),
            ("Q: {question} A:", "{answer}"),
        ]

    def get_question(self, entry):
        question = entry["question"] + "?"
        template = self.get_template(entry)
        return template.replace("{question}", question)

    def get_input_strs(self, entry):
        text = self.get_question(entry)
        return [text] * self.class_num

    def get_answers(self, entry):
        answers =[" " + answer for answer in entry["answer"]]
        return answers

    def get_label(self, entry):
        label = entry["answer"]
        return label

    def get_answer(self, entry):
        return ' '+entry["answer"][0]

# ======================================paraphrase================================
@task_map.add("mrpc")
class Mrpc(BaseTask):
    def __init__(self):
        super().__init__()
        self.class_num = 2
        self.metric = "acc_and_f1"
        self.balance_class = True
        self.cluster = "paraphrase"

    def get_dataset(self, split=None, ds_size=None, cache_dir=None):
        dataset = load_dataset("glue", "mrpc", cache_dir=cache_dir)
        if split == "train":
            return self.load_data_split(dataset, ds_size=ds_size, split=split)
        else:  
            split = "validation"
            return self.load_data_split(dataset, split=split)

    def get_templates(self):
        return [
                ("Here are two sentences: {sentence1} {sentence2} Do they have the same meaning?", "{answer}"),
                ("Here are two sentences: {sentence1} {sentence2} Are the two sentences saying the same thing?", "{answer}"),
                ("{sentence1} {sentence2} Do the above sentences mean the same thing?", "{answer}"),
                ("{sentence1} {sentence2} Please tell me if the sentences above mean the same.", "{answer}"),
                ("{sentence1} {sentence2} Are these sentences conveying the same meaning?", "{answer}"),
                ("{sentence1} {sentence2} If the first sentence is true, is the second one also true?", "{answer}"),
                ("{sentence1} {sentence2} Are these two sentences paraphrases of each other?", "{answer}"),
                ("Do the following two sentences have the same meaning? {sentence1} {sentence2}", "{answer}"),
                ("Do these two sentences mean the same thing? {sentence1} {sentence2}", "{answer}"),
                ("Do these sentences have the same meaning? {sentence1} {sentence2}", "{answer}"),
            ]

    def get_question(self, entry):
        sentence1 = entry["sentence1"]
        sentence2 = entry["sentence2"]
        template = self.get_template(entry)
        return template.replace("{sentence1}", sentence1).replace("{sentence2}", sentence2)

    def get_input_strs(self, entry):
        text = self.get_question(entry)
        return [text] * self.class_num

    def get_answers(self, entry):
        answers = [" No", " Yes"]
        return answers

    def get_label(self, entry):
        label = int(entry["label"])
        return label

    def get_answer(self, entry):
        answers = self.get_answers(entry)
        label = self.get_label(entry)
        return answers[label]


@task_map.add("qqp")
class Qqp(BaseTask):
    def __init__(self):
        super().__init__()
        self.class_num = 2
        self.metric = "acc_and_f1"
        self.balance_class = True
        self.cluster = "paraphrase"

    def get_dataset(self, split=None, ds_size=None, cache_dir=None):
        dataset = load_dataset("glue", "qqp", cache_dir=cache_dir)
        if split == "train":
            return self.load_data_split(dataset, ds_size=ds_size, split=split)
        else:  
            split = "validation"
            return self.load_data_split(dataset, split=split)

    def get_templates(self):
        return [
                ("{question1} {question2} Would you say that these questions are the same?", "{answer}"),
                ("{question1} {question2} Do those questions have the same meaning?", "{answer}"),
                ("{question1} {question2} Are these two questions inquiring about the same information?", "{answer}"),
                ("{question1} {question2} Please tell me if those questions are the same.", "{answer}"),
                ("{question1} {question2} Are these two questions paraphrases of each other?", "{answer}"),
                ("First question: {question1} Second question: {question2} Are these two questions asking the same thing?", "{answer}"),
                ("Question 1: {question1} Question 2: {question2} Are questions 1 and 2 asking the same thing?", "{answer}"),
                ("Question 1: {question1} Question 2: {question2} Would the answer to these two questions be the same?", "{answer}"),
                ("Are the following two questions the same? {question1} {question2}", "{answer}"),
                ("Do these questions have the same meaning? {question1} {question2}", "{answer}"),
            ]

    def get_question(self, entry):
        question1 = entry["question1"].replace('""', "'")
        question2 = entry["question2"].replace('""', "'")
        template = self.get_template(entry)
        return template.replace("{question1}", question1).replace("{question2}", question2)

    def get_input_strs(self, entry):
        text = self.get_question(entry)
        return [text] * self.class_num

    def get_answers(self, entry):
        answers = [" No", " Yes"]
        return answers

    def get_label(self, entry):
        label = int(entry["label"])
        return label

    def get_answer(self, entry):
        answers = self.get_answers(entry)
        label = self.get_label(entry)
        return answers[label]


@task_map.add("paws")
class Paws(BaseTask):
    def __init__(self):
        super().__init__()
        self.class_num = 2
        self.metric = "simple_accuracy"
        self.balance_class = True
        self.cluster = "paraphrase"

    def get_dataset(self, split=None, ds_size=None, cache_dir=None):
        dataset = load_dataset("paws", "labeled_final", cache_dir=cache_dir)
        if split == "train":
            return self.load_data_split(dataset, ds_size=ds_size, split=split)
        else:  
            split = "test"
            return self.load_data_split(dataset, split=split)

    def get_templates(self):
        return [
                ("{sentence1} {sentence2} Do these sentences mean the same thing?", "{answer}"),
                ("{sentence1} {sentence2} Are these two sentences paraphrases of each other?", "{answer}"),
                ("1. {sentence1} 2. {sentence2} Are these two sentences paraphrases of each other?", "{answer}"),
                ("(1) {sentence1} (2) {sentence2} Do these two sentences mean the same thing?", "{answer}"),
                ("Sentence 1: {sentence1} Sentence 2: {sentence2} Do these two sentences convey the same information?", "{answer}"),
                ("Do these two sentences from wikipedia have the same meaning? {sentence1} {sentence2}", "{answer}"),
                ("Same meaning? {sentence1} {sentence2}", "{answer}"),
                ("Are these paraphrases? {sentence1} {sentence2}", "{answer}"),
                ("Do these mean the same? {sentence1} {sentence2}", "{answer}"),
                ("Please check if these have the same meaning. Answer \"yes\" if they do, otherwise \"no\". {sentence1} {sentence2}", "{answer}"),
            ]

    def get_question(self, entry):
        sentence1 = entry["sentence1"]
        sentence2 = entry["sentence2"]
        template = self.get_template(entry)
        return template.replace("{sentence1}", sentence1).replace("{sentence2}", sentence2)

    def get_input_strs(self, entry):
        text = self.get_question(entry)
        return [text] * self.class_num

    def get_answers(self, entry):
        answers = [" No", " Yes"]
        return answers

    def get_label(self, entry):
        label = int(entry["label"])
        return label

    def get_answer(self, entry):
        answers = self.get_answers(entry)
        label = self.get_label(entry)
        return answers[label]


# ================================coreference resolution==================================
@task_map.add("wsc")
class Wsc(BaseTask):
    def __init__(self):
        super().__init__()
        self.class_num = 2
        self.metric = "simple_accuracy"
        self.balance_class = True
        self.cluster = "coreference"

    def get_dataset(self, split=None, ds_size=None, cache_dir=None):
        dataset = load_dataset("super_glue", "wsc", cache_dir=cache_dir)
        if split == "train":
            return self.load_data_split(dataset, ds_size=ds_size, split=split)
        else:  
            split = "validation"
            return self.load_data_split(dataset, split=split)

    def get_templates(self):
        return [
                ("{context} Are \"{text1}\" and \"{text2}\" the same entity?", "{answer}"),
                ("{context} Do \"{text1}\" and \"{text2}\" have the same meaning?", "{answer}"),
                ("Given the following context {context} Are \"{text1}\" and \"{text2}\" the same?", "{answer}"),
                ("{context} Do \"{text2}\" and \"{text1}\" mean the same thing?", "{answer}"),
                ("{context} Are \"{text2}\" and \"{text1}\" the same thing in the aforementioned sentence?", "{answer}"),
                ("Context:{context} Is \"{text2}\" the same as \"{text1}\"?", "{answer}"),
                ("Consider this sentence: {context} Are \"{text2}\" and \"{text1}\" the same?", "{answer}"),
                ("Are \"{text1}\" and \"{text2}\" the same in this sentence? {context}", "{answer}"),
                ("Is \"{text1}\" the same as \"{text2}\" in this sentence? {context}", "{answer}"),
                ("Do \"{text1}\" and \"{text2}\" point to the same thing in the following sentence? {context}", "{answer}"),
            ]

    def get_question(self, entry):
        context = entry["text"]
        text1 = entry["span1_text"]
        text2 = entry["span2_text"]
        template = self.get_template(entry)
        return template.replace("{context}", context).replace("{text1}", text1).replace("{text2}", text2)

    def get_input_strs(self, entry):
        text = self.get_question(entry)
        return [text] * self.class_num

    def get_answers(self, entry):
        answers = [" No", " Yes"]
        return answers

    def get_label(self, entry):
        label = int(entry["label"])
        return label

    def get_answer(self, entry):
        answers = self.get_answers(entry)
        label = self.get_label(entry)
        return answers[label]


@task_map.add("wsc273")
class Wsc273(BaseTask):
    def __init__(self):
        super().__init__()
        self.class_num = 2
        self.metric = "simple_accuracy"
        self.balance_class = False
        self.cluster = "coreference"

    def get_dataset(self, split=None, ds_size=None, cache_dir=None):
        dataset = load_dataset("winograd_wsc", "wsc273", cache_dir=cache_dir)
        if split == "train":
            raise Exception("wsc273 does not have any training set")
        else:  
            split = "test"
            return self.load_data_split(dataset, split=split)

    def get_templates(self):
        return [
                ("{context}", "{answer}"),
                ("Complete the passage. {context}", "{answer}"),
                ("How does this following sentence end? {context}", "{answer}"),
                ("What is the most logical completion for the following text? {context}", "{answer}"),
                ("How does this text end? {context}", "{answer}"),
                ("What happens next? {context}", "{answer}"),
                ("Complete the following sentence. {context}", "{answer}"),
                ("Fill in the remainder of the sentence. {context}", "{answer}"),
                ("What is the next event? {context}", "{answer}"),
                ("Complete the rest of the sentence. {context}", "{answer}"),
            ]

    def get_question(self, entry):
        text_first = entry["text"][: entry["pronoun_loc"]]
        context = text_first
        template = self.get_template(entry)
        return template.replace("{context}", context)

    def get_input_strs(self, entry):
        text = self.get_question(entry)
        return [text + entry["options"][0], text + entry["options"][1]]

    def get_answers(self, entry):
        text_second = entry["text"][entry["pronoun_loc"] + len(entry["pronoun"]):]
        return [text_second] * self.class_num

    def get_label(self, entry):
        label = int(entry["label"])
        return label

    def get_answer(self, entry):  
        text_second = entry["text"][entry["pronoun_loc"] + len(entry["pronoun"]):]
        answers = [entry["options"][0] + text_second, entry["options"][1] + text_second]
        label = self.get_label(entry)
        return answers[label]


@task_map.add("winogrande")
class Winogrande(BaseTask):
    def __init__(self):
        super().__init__()
        self.class_num = 2
        self.metric = "simple_accuracy"
        self.balance_class = False
        self.cluster = "coreference"

    def get_dataset(self, split=None, ds_size=None, cache_dir=None):
        dataset = load_dataset("winogrande", "winogrande_xl", cache_dir=cache_dir)
        if split == "train":
            return self.load_data_split(dataset, ds_size=ds_size, split=split)
        else:  
            split = "validation"
            return self.load_data_split(dataset, split=split)

    def get_templates(self):
        return [
                ("How does the sentence end? {context}", "{answer}"),
                ("Write the next sentence. {context}", "{answer}"),
                ("Continue the following story. {context}", "{answer}"),
                ("Complete the following sentence. {context}", "{answer}"),
                ("Continue writing the following text. {context}", "{answer}"),
                ("How does the sentence end? {context}", "{answer}"),
                ("Write the next sentence. {context}", "{answer}"),
                ("Continue the following story. {context}", "{answer}"),
                ("Complete the following sentence. {context}", "{answer}"),
                ("Continue writing the following text. {context}", "{answer}"),
            ]

    def get_question(self, entry):
        cut_index = entry["sentence"].index("_")
        context = entry["sentence"][:cut_index]
        template = self.get_template(entry)
        return template.replace("{context}", context)

    def get_input_strs(self, entry):
        text = self.get_question(entry)
        return [text + entry["option1"], text + entry["option2"]]

    def get_answers(self, entry):
        cut_index = entry["sentence"].index("_")
        text_second = entry["sentence"][cut_index + 1 :]
        answers = [text_second] * self.class_num
        return answers

    def get_label(self, entry):
        label = int(entry["answer"]) - 1
        return label

    def get_answer(self, entry):
        cut_index = entry["sentence"].index("_")
        text_second = entry["sentence"][cut_index + 1 :]
        answers = [entry["option1"] + text_second, entry["option2"] + text_second]
        label = self.get_label(entry)
        return answers[label]


# ===============================Struct-to-Text=======================
@task_map.add("common_gen")
class Common_gen(BaseTask):
    def __init__(self):
        super().__init__()
        self.class_num = 1
        self.metric = "rouge"
        self.balance_class = False
        self.cluster = "struct2text"
        self.finder_L = 100
        self.run_scorer_bsz = 10

    def get_dataset(self, split=None, ds_size=None, cache_dir=None):
        dataset = load_dataset("common_gen", cache_dir=cache_dir)
        if split == "train":
            return self.load_data_split(dataset, ds_size=ds_size, split=split)
        else:  
            split = "validation"
            return self.load_data_split(dataset, split=split)

    def get_templates(self):
        return [
                ("Concepts: {concepts}. Write a sentence that includes all these words.", "{target}"),
                ("Keywords: {concepts}. What is a sentence that includes all these keywords?", "{target}"),
                ("Here are some concepts: {concepts}. What is a sentence about these concepts?", "{target}"),
                ("Produce a sentence which mentions all of these concepts: {concepts}.", "{target}"),
                ("Write a sentence about the following things: {concepts}.", "{target}"),
                ("Generate a sentence that includes all the following words: {concepts}.", "{target}"),
            ]

    def get_question(self, entry):
        concepts = ", ".join(entry["concepts"])
        template = self.get_template(entry)
        return template.replace("{concepts}", concepts)

    def get_input_strs(self, entry):
        text = self.get_question(entry)
        return [text]

    def get_answers(self, entry):
        answers = [" "+entry["target"]]
        return answers

    def get_label(self, entry):
        return entry["target"]

    def get_answer(self, entry):  
        return " "+entry["target"]


@task_map.add("dart")
class Dart(BaseTask):
    def __init__(self):
        super().__init__()
        self.class_num = 1
        self.metric = "rouge"
        self.balance_class = False
        self.cluster = "struct2text"

    def get_dataset(self, split=None, ds_size=None, cache_dir=None):
        dataset = load_dataset("GEM/dart", cache_dir=cache_dir)
        if split == "train":
            return self.load_data_split(dataset, ds_size=ds_size, split=split)
        else:  
            split = "validation"
            return self.load_data_split(dataset, split=split)

    def get_templates(self):
        return [
                ("Triple: {tripleset} What is a sentence that describes this triple?", "{target}"),
                ("Data: {tripleset} What would a sentence about this data be like?", "{target}"),
                ("Generate an approximately fifteen-word sentence that describes all this data: {tripleset}", "{target}"),
                ("Here is some data: {tripleset}. Write a sentence that describes this data", "{target}"),
                ("This is some data: {tripleset}. Generate a detailed description of this data", "{target}"),
                ("Generate a sentence about this data: {tripleset}", "{target}"),
                ("Write a sentence that about [{tripleset}].", "{target}"),
                ("Produce a long descriptive sentence that uses all these words: {tripleset}", "{target}"),
            ]

    def get_question(self, entry):
        tripleset = "; ".join([", ".join(triplet) for triplet in entry["tripleset"]])
        # Get rid of some undesirable cells like "[TABLECONTEXT]", "[TITLE]"
        tripleset = re.sub(r"\[(.*?)\]", "", tripleset)
        template = self.get_template(entry)
        return template.replace("{tripleset}", tripleset)

    def get_input_strs(self, entry):
        text = self.get_question(entry)
        return [text]

    def get_answers(self, entry):
        answers = [" " + entry["target"]]
        return answers

    def get_label(self, entry):
        return entry["target"]

    def get_answer(self, entry):
        return " "+entry["target"]


@task_map.add("e2e_nlg")
class E2e_nlg(BaseTask):
    def __init__(self):
        super().__init__()
        self.class_num = 1
        self.metric = "rouge"
        self.balance_class = False
        self.cluster = "struct2text"

    def get_dataset(self, split=None, ds_size=None, cache_dir=None):
        dataset = load_dataset("GEM/e2e_nlg", cache_dir=cache_dir)
        if split == "train":
            return self.load_data_split(dataset, ds_size=ds_size, split=split)
        else:  
            split = "test"
            return self.load_data_split(dataset, split=split)

    def get_templates(self):
        return [
                ("Attributes: {meaning_representation}. Produce a detailed sentence about this restaurant.", "{target}"),
                ("Data: {meaning_representation}. Can you generate a sentence about this data?", "{target}"),
                ("Data: {meaning_representation}. What is a sentence that describe this data?", "{target}"),
                ("Here are some keywords about a restaurant: {meaning_representation}. Write a sentence that describes the following attributes of a restaurant.", "{target}"),
                ("Here is some data about a restaurant: {meaning_representation}. Write a sentence that includes the following data about a restaurant.", "{target}"),
                ("Sentence: {meaning_representation}. Can you represent the content in this sentence in data form?", "{target}"),
                ("Write a sentence about a restaurant with all the following attributes: {meaning_representation}.", "{target}"),
                ("Write a sentence that is about a restaurant with all the following properties: {meaning_representation}.", "{target}"),
                ("Produce a detailed sentence about a restaurant using the following words: {meaning_representation}.", "{target}"),
                ("Generate a descriptive sentence about a restaurant using the following words: {meaning_representation}.", "{target}"),
            ]

    def get_question(self, entry):
        meaning_representation = re.sub(r"\[", " = ", entry["meaning_representation"])
        meaning_representation = re.sub(r"\]", "", meaning_representation)
        template = self.get_template(entry)
        return template.replace("{meaning_representation}", meaning_representation)

    def get_input_strs(self, entry):
        text = self.get_question(entry)
        return [text]

    def get_answers(self, entry):
        answers = [" " + entry["target"]]
        return answers

    def get_label(self, entry):
        return entry["target"]

    def get_answer(self, entry):
        return " " + entry["target"]


# ===============================Summarization============================
@task_map.add("ag_news")
class Ag_news(BaseTask):
    def __init__(self):
        super().__init__()
        self.class_num = 4
        self.metric = "simple_accuracy"
        self.balance_class = True
        self.cluster = "summarize"

    def get_dataset(self, split=None, ds_size=None, cache_dir=None):
        dataset = load_dataset("ag_news", cache_dir=cache_dir)
        if split == "train":
            return self.load_data_split(dataset, ds_size=ds_size, split=split)
        else:  
            split = "test"
            return self.load_data_split(dataset, split=split)

    def get_templates(self):
        return [
                ("\"{text}\" What is this text about? World, Sports, Business, or Technology?", "{answer}"),
                ("\"{text}\" Which topic is this article about? World, Sports, Business, or Technology?", "{answer}"),
                ("\"{text}\" Which is the best summary of this article? World, Sports, Business, or Technology?", "{answer}"),
                ("\"{text}\" What is this text about? World, Sports, Business, or Technology?", "{answer}"),
                ("\"{text}\" What best summarizes the content of the above article? World, Sports, Business, or Technology?", "{answer}"),
                ("Which is this about? \"{text}\" World, Sports, Business, or Technology?", "{answer}"),
                ("Which is an appropriate title for this article? \"{text}\" World, Sports, Business, or Technology?", "{answer}"),
                ("Select the topic that this about: \"{text}\" World, Sports, Business, or Technology?", "{answer}"),
            ]

    def get_question(self, entry):
        text = entry["text"]
        template = self.get_template(entry)
        return template.replace("{text}", text)

    def get_input_strs(self, entry):
        text = self.get_question(entry)
        return [text] * self.class_num

    def get_answers(self, entry):
        answers = [" World", " Sports", " Business", " Technology"]
        return answers

    def get_label(self, entry):
        return int(entry["label"])

    def get_answer(self, entry):
        answers = self.get_answers(entry)
        label = self.get_label(entry)
        return answers[label]


@task_map.add("aeslc")
class Aeslc(BaseTask):
    def __init__(self):
        super().__init__()
        self.class_num = 1
        self.metric = "rouge"
        self.balance_class = False
        self.cluster = "summarize"
        self.run_scorer_bsz = 10
        self.finder_L = 50

    def filter(self, entry):
        text = entry["email_body"]
        text = re.sub(r"\n", " ", text)
        answer = entry["subject_line"]
        answer = re.sub(r"\n", " ", answer)
        return (
            len(text.split()) > 0
            and len(answer.split()) > 0
            and len(text.split()) <= 256
            and len(answer.split()) <= 256
        )

    def get_dataset(self, split=None, ds_size=None, cache_dir=None):
        dataset = load_dataset("aeslc", cache_dir=cache_dir)
        if split == "train":
            return self.load_data_split(dataset, ds_size=ds_size, split=split)
        else:  
            split = "test"
            return self.load_data_split(dataset, split=split)

    def get_templates(self):
        return [
                ("What is the subject line for this email? {body}", "{subject}"),
                ("Write a subject line for this message: {body}", "{subject}"),
                ("{body} Write a subject line for this email.", "{subject}"),
                ("Here is an email: {body} What is a potential subject line for this email?", "{subject}"),
                ("{body} Propose a subject line for this email?", "{subject}"),
                ("This is the content of an email: {body} What was the subject line for this email?", "{subject}"),
                ("This is an email {body} What is the subject of this email?", "{subject}"),
                ("{body} Generate a subject line for this email.", "{subject}"),
            ]

    def get_question(self, entry):
        body = re.sub(r"\n", " ", entry["email_body"])
        template = self.get_template(entry)
        return template.replace("{body}", body)

    def get_input_strs(self, entry):
        text = self.get_question(entry)
        return [text] * self.class_num

    def get_answers(self, entry):
        subject = re.sub(r"\n", "", entry["subject_line"])
        answers = [" " + subject]
        return answers

    def get_label(self, entry):
        return re.sub(r"\n", "", entry["subject_line"])

    def get_answer(self, entry):
        return " " + re.sub(r"\n", "", entry["subject_line"])


@task_map.add("gigaword")
class Gigaword(BaseTask):
    def __init__(self):
        super().__init__()
        self.class_num = 1
        self.metric = "rouge"
        self.balance_class = False
        self.cluster = "summarize"
        self.run_scorer_bsz = 20
        self.finder_L = 100

    def filter(self, entry):
        text = "".join([entry["document"], entry["summary"]])
        no_unk = "UNK" not in text
        no_hashtag = "#" not in text
        return no_unk and no_hashtag

    def get_dataset(self, split=None, ds_size=None, cache_dir=None):
        dataset = load_dataset("gigaword", cache_dir=cache_dir)
        if split == "train":
            return self.load_data_split(dataset, ds_size=ds_size, split=split)
        else:  
            split = "test"
            return self.load_data_split(dataset, split=split)

    def get_templates(self):
        return [
                ("Write a short summary for this text: {text}", "{summary}"),
                ("Briefly summarize this sentence: {text}", "{summary}"),
                ("Generate a short summary this sentence: {text}", "{summary}"),
                ("What is a shorter version of this: {text}", "{summary}"),
                ("{text} Write a brief summary in a sentence or less", "{summary}"),
                ("{text} What is a very short summary of the above text?", "{summary}"),
                ("{text} Summarize the aforementioned text in a single phrase.", "{summary}"),
                ("{text} Can you generate a short summary of the above paragraph?", "{summary}"),
            ]

    def get_question(self, entry):
        text = entry["document"]
        template = self.get_template(entry)
        return template.replace("{text}", text)

    def get_input_strs(self, entry):
        text = self.get_question(entry)
        return [text] * self.class_num

    def get_answers(self, entry):
        answers = [" " + entry["summary"]]
        return answers

    def get_label(self, entry):
        return entry["summary"]

    def get_answer(self, entry):
        return " " + entry["summary"]

# ========================== Exploration Example: Chain-of-Thought Prompting ========================
# define your cot prompting task
@task_map.add("pubmed_qa")
class Pubmed_qa(BaseTask):
    def __init__(self):
        super().__init__()
        self.class_num = 1 # we regard cot as a text completion task
        self.metric = "pubmed_qa_acc"
        self.cluster = "reading"

    # get dataset splits
    def get_dataset(self, split=None, ds_size=None, cache_dir=None):
        dataset = load_dataset("pubmed_qa", 'pqa_labeled', cache_dir=cache_dir)
        # pubmed_qa has no validation splits, we create our own random evaluations splits
        split_ratio = 0.8
        data = list(dataset['train'].shuffle(seed=42))
        if split == "train":
            return data[:int(len(data)*split_ratio)]
        else:  
            return data[int(len(data)*split_ratio):]

    # define cot task templates to transfer the datasets to instructions, 
    # we use the templates of `synth_cot_cosmos_qa` in FLANv2: https://github.com/google-research/FLAN/blob/main/flan/v2/templates.py
    # Remove the newline character for better prompting performance (especially for small LLMs)
    def get_templates(self):
        return [
            ("{context} Question: {question} Yes, No, or Maybe? Let's answer step by step.", "{cot} So the answer is {answer}"),
            ("{context} Q: {question} Yes, No, or Maybe? Step by step reasoning:", "{cot} The answer is {answer}"),
            ("{context} Let's answer this carefully: {question} Yes, No, or Maybe?", "{cot} The answer is {answer}"),
            ("{context} Based on the preceding passage, answer question {question} Yes, No, or Maybe? Let's solve slowly:", "{cot} The answer is {answer}"),
            ("{context} Solve the following question thinking out loud: {question} Yes, No, or Maybe?", "{cot} So, the answer is {answer}"),
            ("Context: {context} Question: {question} Yes, No, or Maybe? Let's think:", "{cot}... So the answer is {answer}"),
            ("Read the following article and answer the question. {context} {question} Yes, No, or Maybe? ... Chain-of-thought:", "{cot} The answer is {answer}"),
            ("Answer the question about text: {context} {question} Yes, No, or Maybe? CoT:", "{cot} The answer is {answer}"),
            ("{context} Question: {question} Yes, No, or Maybe? Chain-of-thought:", "{cot} The answer is {answer}"),
            ("Context: {context} Q: {question} Yes, No, or Maybe? Step-by-step reasoning process:", "{cot} The answer is {answer}"),
            ]

    # random_sample one template to convert the task input to an instruction
    def get_question(self, entry):
        question = entry["question"]
        meta_context = entry["context"]
        contexts=[]
        for i, label in enumerate(meta_context["labels"]):
            content = meta_context["contexts"][i]
            sub_title = label[0]+label[1:].lower()
            contexts.append(f'({sub_title}) {content}')
        context = '\n'.join(contexts)
        question_template = self.get_template(entry, return_answer=False)
        return question_template.replace("{context}", context).replace("{question}", question)

    # wrap the question as a list for scoring/inference, align with mutiple choice task
    def get_input_strs(self, entry):
        text = self.get_question(entry)
        return [text]

    # wrap answer as a list for scoring/inference
    def get_answers(self, entry):
        cot = entry["long_answer"]
        answer = entry["final_decision"]

        # we fix the random seed as the entry["id"] when sampling the question and answer templates,
        # thus the question and answer templates always correspond to each other
        answer_template = self.get_template(entry, return_answer=True)
        answer_template = answer_template.replace('{cot}', cot).replace('{answer}', answer)
        answers = [' ' + answer_template]
        return answers

    # get label completion(s) for calculating cot acc
    def get_label(self, entry):
        label = entry["final_decision"]
        return label

    # the get_answer function is for constructing demonstration in the prompt pool, we return a string
    def get_answer(self, entry):
        cot = entry["long_answer"]
        answer = entry["final_decision"]

        answer_template = self.get_template(entry, return_answer=True)
        return ' ' + answer_template.replace('{cot}', cot).replace('{answer}', answer)
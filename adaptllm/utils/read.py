from concurrent.futures import ProcessPoolExecutor
import random
import os
import sentencepiece as spm
import numpy as np
import re

TYPES=['nli', 'common_reason', 'paraphrase', 'word2text', 'summarize', 'text_completion']

def remove_double_space(string):
    return re.sub("[ ]{2,}", " ", string)

def get_max_workers():
     # create a process pool with the default number of worker processes
    pool = ProcessPoolExecutor()
    # report the number of worker processes chosen by default
    return pool._max_workers

class App:
    def __init__(self):
        self.cls_dic = {}

    def add(self, key):
        def adder(cls):
            self.cls_dic[key] = cls
            return cls
        return adder

type_map = App()
# usage: import type_map; cls=type_map.cls_dic[type_name]()

@type_map.add("basetype")
class BaseType(object):
    def __init__(self):
        # NOTE: we use a simple blank space to connect input and output
        # you may try other connectors like `<eot>` in LIMA or `## Response:` in Orca
        self.qa_deliminator = ' '
        self.max_subcategory_num = 2 # limit the number of examples per subcategory
        self.max_seq_len = 2048

    def get_template(self, entry, random_seed):
        '''
        random sample a template for each entry
        '''
        random.seed(random_seed) # fix random seed for reproduction
        template = random.choice(self.get_all_templates(entry, random_seed))
        return template

    def fill_in_the_template(self, template, kw_dic):
        question = template[0].format(**kw_dic)
        answer = template[1].format(**kw_dic)
        return question + self.qa_deliminator + answer
    
    def truncate_sentence(self, text, max_len):
        tokenized_example = self.ori_spm.encode(text)
        example_length = len(tokenized_example)
        
        if example_length > max_len:
            input_ids = tokenized_example[:max_len]
            truncated_text = self.ori_spm.decode(input_ids)
            return truncated_text
        else:
            return text
    
    def init_spm(self, ori_spm_path, domain_spm_path):
        self.ori_spm = spm.SentencePieceProcessor(model_file=ori_spm_path)
        ori_tokens = set([self.ori_spm.id_to_piece(i) for i in range(len(self.ori_spm))])

        self.domain_spm = spm.SentencePieceProcessor(model_file=domain_spm_path)
        domain_tokens = set([self.domain_spm.id_to_piece(i) for i in range(len(self.domain_spm))])
        specific_tokens = domain_tokens-(ori_tokens & domain_tokens)
        specific_tokens = [token for token in list(specific_tokens) if (token.startswith('▁') and len(token)>10)]
        self.specific_token_set = set(specific_tokens)

    def compile_regex(self):
        self.regex_dic={}
        for class_name, pattern in self.mine_regex.items():
            self.regex_dic[class_name] = re.compile(pattern, re.IGNORECASE)
    
    def mine(self, text, **kwargs):
        mined_dic = {}
        mined_num = 0
        for class_name, regex in self.regex_dic.items(): 
            mined_dic[class_name]=[]
            x = regex.findall(text)
            if len(x)>0:
                for tup in x:
                    collected = self.collect_mined(tup, class_name)
                    mined_dic[class_name].append(collected)   
            mined_num += len(mined_dic[class_name])
        return mined_dic, mined_num


@type_map.add("nli")
class nli(BaseType):
    def __init__(self, ori_spm_path = None, domain_spm_path = None):
        super().__init__()
        # init regex
        self.mine_regex = {
		'Entail': r'([.!?]+[\s]+)([^.!?\n,]{50,}[.!?]+)([\s]+)(Yes|Therefore|Thus|Accordingly|Hence|For this reason)([\s]*,[\s]+)([^.!?\n,]{50,}[.!?]+)([\s]+)',
        'Contradict': r'([.!?]+[\s]+)([^.!?\n,]{50,}[.!?]+)([\s]+)(No|However|But|On the contrary|In contrast|Whereas)([\s]*,[\s]+)([^.!?\n,]{50,}[.!?]+)([\s]+)',
		'Neutral': r'([.!?]+[\s]+)([^.!?\n,]{50,}[.!?]+)([\s]+)(Maybe|Also|Furthermore|Secondly|Additionally|Moreover|In addition)([\s]*,[\s]+)([^.!?\n,]{50,}[.!?]+)([\s]+)'
	    }
        self.compile_regex() 

    def collect_mined(self, tup, class_name):
        dic = {
            'label': class_name,
            'verbalizer': tup[3],
            'premise': tup[1],
            'hypothesis': tup[-2],
        }
        return dic

    def get_all_templates(self, entry, random_seed):
        np.random.seed(random_seed)
        type = np.random.choice(['generate','classify'], p=[0.2, 0.8])
        if type == 'classify':
            return [
            # Basic Templates
            ("{premise}\nBased on the sentence above can we infer that \"{hypothesis}\"?", "{answer}"),
            ("{premise}\nBased on this sentence can we infer that the following sentence is true?\n{hypothesis}\nAnswer:", "{answer}"),
            ("{premise}\nCan we draw the following hypothesis?\n{hypothesis}\n{options_}", "{answer}"),
            ("{premise}\nDoes this next sentence follow, given the preceding text?\n{hypothesis}\nAnswer:", "{answer}"),
            ("Can we draw the following hypothesis from the context?\nContext: {premise}\nHypothesis: {hypothesis}\nAnswer:", "{answer}"),
            ("{hypothesis}\nDetermine if the sentence is true based on the text below:\n{premise}\nAnswer:", "{answer}"),
            ("Premise: {premise}\nHypothesis: {hypothesis}\nDoes the premise entail the hypothesis?", "{answer}"),
            ("Premise: {premise}\nHypothesis: {hypothesis}\nIs the hypothesis entailed by the premise?", "{answer}"),
            ("Here is a premise:\n{premise}\nHere is a hypothesis:\n{hypothesis}\nIs it possible to infer that if the premise is true, then so is the hypothesis?", "{answer}"),
            ("Sentence 1: {premise}\nSentence 2: {hypothesis}\nIs this second sentence entailed by the first sentence?\n{options_}", "{answer}"),
            ("Based on the premise \"{premise}\", can we infer the hypothesis \"{hypothesis}\" is true?", "{answer}"),
            ("Premise:\n\"{premise}\" Based on this premise, is the hypothesis \"{hypothesis}\" true?\n{options_}", "{answer}"),
            ("If {premise}, can we infer that \"{hypothesis}\"?", "{answer}"),
            ("{premise}\nDoes it follow that \"{hypothesis}\"?\n{options_}", "{answer}"),
            ("Question: If \"{premise}\", does this mean that \"{hypothesis}\"?\nAnswer:", "{answer}"),
            ("If \"{premise}\", can we infer \"{hypothesis}\"?", "{answer}"),
            ("If \"{premise}\", does it logically follow that \"{hypothesis}\"?", "{answer}"),
            ("Based on the sentence \"{premise}\", is the sentence \"{hypothesis}\" a true sentence?", "{answer}"),
            ("Premise: {premise}\nHypothesis: {hypothesis}\nCan we infer that the hypothesis is true if the premise is true?", "{answer}"),
            ("Here is a premise: \"{premise}\"\nHere is a hypothesis: \"{hypothesis}\"\nDoes the premise tell us whether the hypothesis is true?", "{answer}"),
            ("Is the premise \"{premise}\" true if \"{hypothesis}\"?\n{options_}", "{answer}"),
            ("If \"{premise}\", can we infer that \"{hypothesis}\"?\n{options_}", "{answer}"),
            ("If \"{premise}\", is \"{hypothesis}\" correct?", "{answer}"),
            ("Let's say that \"{premise}\"\nCan we now say that \"{hypothesis}\"?", "{answer}"),
            ("Does \"{hypothesis}\" appear to be an accurate statement based on \"{premise}\"?", "{answer}"),
            ("Is it possible to draw the statement that \"{hypothesis}\" if \"{premise}\"?", "{answer}"),
            ("Is \"{hypothesis}\" true if \"{premise}\"?\n{options_}", "{answer}"),
            ("Sentence 1: \"{premise}\"\nSentence 2: \"{hypothesis}\"\nIs sentence 2 true, based on sentence 1?", "{answer}"),

            # fill-in-the-blank:
            ("Sentence 1: \"{premise}\"\nSentence 2: \"{hypothesis}\"\nWhich word is the best to connect them? Therefore, However, or Moreover?", "{connect_answer}"),
            ("Choose the most suitable word to link the following sentences:\n1. {premise}\n2. {hypothesis}\nOptions:\n- Therefore\n- However\n- Moreover", "{connect_answer}"),
            ("Connect the following sentence: {premise}\nChoose the appropriate word to link it with: \"{hypothesis}\"\nOptions: Therefore, However, Moreover", "{connect_answer}"),
            ("Given the sentence: {premise}\nChoose the appropriate word from the options (Therefore, However, Moreover) to connect it with: \"{hypothesis}\"\nWord:", "{connect_answer}"),
            ("Connect the sentence: {premise}\nFrom the choices (Therefore, However, Moreover), select the word that best links it to: \"{hypothesis}\"\nAnswer:", "{connect_answer}"),
            
            # relation classification
            ("Assess the relationship between Sentence 1: \"{premise}\"\nSentence 2: \"{hypothesis}\"\nIs it characterized as Entailment, Neutral, or Contradictory?", "{relation_answer}"),
            ("Given Sentence 1: \"{premise}\"\nSentence 2: \"{hypothesis}\"\nHow would you describe the relationship between these two sentences? Entailment, Neutral, or Contradictory?", "{relation_answer}"),
            ("Considering Sentence 1: \"{premise}\"\nSentence 2: \"{hypothesis}\"\nHow do you perceive the connection between these two sentences in terms of their relationship?", "{relation_answer}"),
            ("Assess the relationship between Sentence 1: \"{premise}\"\nSentence 2: \"{hypothesis}\"\nWould you categorize their connection as Entailment, Neutral, or Contradictory?", "{relation_answer}"),
        ] 
        elif type == 'generate':
            if entry['label'] == 'Entail':
                return [
            ('Complete the following sentence\n{premise} Accordingly,', "{hypothesis}"),
            ('{premise} Therefore:', "{hypothesis}"),
            ('{premise} Thus?', "{hypothesis}"),
            ("Based on the statement \"{premise}\", provide a continuation using the word \"Hence\" to express the following idea.\nContinuation:", "{hypothesis}"),
            ("Question: Complete the following statement using the word \"Therefore\" in relation to \"{premise}\"\nAnswer:", "{hypothesis}"),
            ("{premise} {verbalizer}?", "{hypothesis}"),
            ("{premise} {verbalizer}:", "{hypothesis}"),

            # more variations
            ("{premise}\nProduce a sentence that encompasses the concept from the above statement. Sentence:", "{hypothesis}"),
            ("\"{premise}\" Generate a sentence that follows from the notion presented in the previous statement.", "{hypothesis}"),
            ("{premise}\nCraft a sentence that develops the idea put forth in the preceding statement.", "{hypothesis}"),
            ("{premise}\nCreate a sentence that is a logical extension of the idea in the previous statement.\nAnswer:", "{hypothesis}"),
            ("\"{premise}\" Formulate a sentence that is consistent with the concept presented in the prior statement.", "{hypothesis}"),
            ("{premise}\nDevelop a sentence that builds upon the thought conveyed in the above statement.", "{hypothesis}"),
            ]
            elif entry['label'] == 'Neutral':
                return [
            ('Complete the following sentence: {premise} {verbalizer},', "{hypothesis}"),
            ('Complete the following sentence\n{premise} {verbalizer}:', "{hypothesis}"),
            ('{premise} {verbalizer}?', "{hypothesis}"),
            ("Based on the statement {premise}, provide a continuation using the word \"{verbalizer}\" to express the following idea.\nContinuation:", "{hypothesis}"),
            ("Question: Complete the following statement using the word \"{verbalizer}\" in relation to \"{premise}\"\nAnswer:", "{hypothesis}"),
            ]
            elif entry['label'] == 'Contradict':
                return [
            ('Complete the following sentence: {premise} On the contrary,', "{hypothesis}"),
            ('{premise} But,\nWhat is a completion for it?', "{hypothesis}"),
            ('Complete the following sentence\n{premise} However?', "{hypothesis}"),
            ("Sentence: {premise} {verbalizer},\nHow do you finish this sentence?", "{hypothesis}"),
            ("{premise} {verbalizer}:", "{hypothesis}"),
            ("Based on the statement {premise}, provide a continuation using \"In contrast\" to express the following idea.", "{hypothesis}"),
            ("Complete the following statement using the word \"But\" in relation to \"{premise}\".", "{hypothesis}"),
            ]

    def format_single_demo(self, entry, random_seed):
        kw_dic = {}
        kw_dic['premise'] = entry["premise"]
        hypothesis = entry["hypothesis"]
        kw_dic['hypothesis'] = hypothesis[0].upper()+hypothesis[1:]
        kw_dic['options_'] = '- Yes\n- No\n- Maybe'

        kw_dic['verbalizer'] =entry['verbalizer']
        if entry['label'] == 'Entail':
            kw_dic['answer'] = 'Yes'
            kw_dic['connect_answer'] = 'Therefore'
            kw_dic['relation_answer'] = 'Entailment'
        elif entry['label'] == 'Contradict':
            kw_dic['answer'] = 'No'
            kw_dic['connect_answer'] = 'However'
            kw_dic['relation_answer'] = 'Contradictory'
        elif entry['label'] == 'Neutral':
            kw_dic['answer'] = 'Maybe'
            kw_dic['connect_answer'] = 'Moreover'
            kw_dic['relation_answer'] = 'Neutral'

        template = self.get_template(entry, random_seed)
        return self.fill_in_the_template(template, kw_dic)      

@type_map.add("common_reason")
class common_reason(BaseType):
    def __init__(self):
        super().__init__()
        self.mine_regex = {
        'Cause-effect': r'([.!?]+[\s]+)([^.!?\n,]{50,}[.!?]+)([\s]+)(Thus|Therefore|Accordingly|Hence|For this reason)([\s]*,[\s]+)([^.!?\n,]{50,}[.!?]+)([\s]+)',
        'Effect-cause':r'([.!?]+[\s]+)([^.!?;\n,]{50,}[.!?]+)([\s]+)(due to|on account of|owing to)([\s]+)([^.!?;\n,]{50,}[.!?]+)([\s]+)',
        }
        self.compile_regex()
    
    def collect_mined(self, tup, class_name):
        dic = {
            'relation': class_name,
            'verbalizer': tup[3],
            'sentence1': tup[1],
            'sentence2': tup[-2],
        }
        return dic
    
    def get_all_templates(self, entry, random_seed):
        if entry['relation'] == 'Cause-effect':
            return [
            # Basic templates
            ("Question: What is the effect of \"{cause}\"? Answer:", "{effect}"),
            ("Here is a premise: {cause}\nWhat is the effect?", "{effect}"),
            ("Q: What is the result of \"{cause}\"? A:", "{effect}"),
            ("What is a plausible effect of \"{cause}\"?", "{effect}"),
            ("Based on \"{cause}\", what is the result?", "{effect}"),
            ("{cause}\nEffect:", "{effect}"),
            ("What is the result of the following sentence?\n{cause}\nResult:", "{effect}"),
            ("Q: What happens after \"{cause}\"? A:", "{effect}"),
            ("{cause}\nWhat happens next?", "{effect}"),

            # More varaiations
            ("Considering the cause: {cause}\nWhat could be the resulting effect?", "{effect}"),
            ("Given that: {cause}\nWhat do you anticipate as the outcome?", "{effect}"),
            ("What could stem from \"{cause}\"?", "{effect}"),
            ("Explore the consequences of: {cause}\nAnswer:", "{effect}"),
            ("What might follow from \"{cause}\"?", "{effect}"),
            ("Based on the cause: \"{cause}\"\nWhat is likely to be the effect?", "{effect}"),
            ("If \"{cause}\" occurs, what is the probable effect?", "{effect}"),
            ("Imagine \"{cause}\" taking place; what would be the resultant effect?", "{effect}"),
            ("Given the scenario: {cause}\nWhat effect could be expected?", "{effect}"),
            ("Examine the potential outcomes of \"{cause}\"\nOutcome:", "{effect}"),
            ("Anticipating the result of: {cause}\nWhat could be the effect?", "{effect}"),
            ("What is the expected effect of \"{cause}\"?", "{effect}"),
            ("Considering the event: {cause}\nWhat could be an outcome?", "{effect}"),
            ("If \"{cause}\" happens, what could be the subsequent effect?", "{effect}"),
            ("Explore the aftermath of: \"{cause}\"\nWhat could be the effect?", "{effect}"),
            ]
        elif entry['relation'] == 'Effect-cause':
            return [
            # Basic templates
            ("Q: \"{effect}\" What is the cause? A:", "{cause}"),
            ("Here is a result: {effect}\nWhat is the cause?", "{cause}"),
            ("What is the reason of \"{effect}\"?", "{cause}"),
            ("What is a plausible reason for \"{effect}\"?", "{cause}"),
            ("what is the cause of \"{effect}\"?", "{cause}"),
            ("{effect}\nCause:", "{cause}"),
            ("Question: What is the reason of the following sentence?\n{effect}\nAnswer:", "{cause}"),
            ("What happens before \"{effect}\"?", "{cause}"),
            ("{effect}\nWhat happens before?", "{cause}"),

            # More variations:
            ("Given the outcome: {effect}\nWhat could have led to this result?", "{cause}"),
            ("Uncover the cause behind: \"{effect}\".", "{cause}"),
            ("What might be responsible for {effect}?", "{cause}"),
            ("Identify a probable cause for: {effect}\nCause:", "{cause}"),
            ("What event or circumstance could explain \"{effect}\"?", "{cause}"),
            ("When observing: {effect}\nWhat should we consider as the cause?", "{cause}"),
            ("What events or factors contributed to: {effect}?", "{cause}"),
            ("Considering the effect: \"{effect}\"\nWhat could be the underlying cause?", "{cause}"),
            ("Before \"{effect}\" occurred, what factor might have caused it?", "{cause}"),
            ("What do you think led to the occurrence of: \"{effect}\"?", "{cause}"),
            ("Analyze the occurrence of: {effect}\nWhat could be identified as the cause?", "{cause}"),
            ("Given that: {effect}\nWhat was the triggering cause?", "{cause}"),
            ("Explore the background of: \"{effect}\"\nWhat could have instigated it?", "{cause}"),
            ("What played a role in bringing about: {effect}?", "{cause}"),
            ("Delve into the circumstances behind \"{effect}\"\nWhat could be the originating cause? Answer:", "{cause}"),         
            ('Complete the following sentence\n{effect} because of', "{cause}"),
            ('Your task is to complete the following sentence: {effect} due to', "{cause}"),
            ('{effect} owing to\nHow would you complete it:', "{cause}"),
            ("Based on the statement {effect}, provide a continuation using \"{verbalizer}\" to express the following idea.\nContinuation:", "{cause}"),
            ("Question: Complete the following statement using \"{verbalizer}\" in relation to \"{effect}\".", "{cause}"),
            ("Answer the question...{effect} {verbalizer}?", "{cause}"),
            ("{effect} {verbalizer}:", "{cause}"),
        ]
    def format_single_demo(self, entry, random_seed):
        kw_dic = {}
        kw_dic['verbalizer'] = entry['verbalizer']
        if entry['relation'] == 'Cause-effect':
            kw_dic['cause'] =  entry['sentence1']
            kw_dic['effect'] = entry['sentence2'][0].upper() + entry['sentence2'][1:]
        elif entry['relation'] == 'Effect-cause':
            kw_dic['cause'] = entry['sentence2'][0].upper() + entry['sentence2'][1:]
            kw_dic['effect'] = entry['sentence1']
        elif entry['relation'] == 'Explanantion':
            kw_dic['sentence1'] = entry['sentence1']
            kw_dic['sentence2'] = entry['sentence2'][0].upper() + entry['sentence2'][1:]
        
        template = self.get_template(entry, random_seed)
        return self.fill_in_the_template(template, kw_dic)

@type_map.add("paraphrase")
class paraphrase(BaseType):
    def __init__(self):
        super().__init__()
        self.mine_regex = {
        'Similar': r'([.!?]+[\s]+)([^.!?\n]{50,}[.!?]+)([\s]+)(In other words|In other word|Namely|That is to say|i.e.|Scilicet|Similarly|Equally)([\s]*,[\s]+)([^.!?\n]{50,}[.!?]+)([\s]+)',
        'Different': r'([.!?]+[\s]+)([^.!?\n]{50,}[.!?]+)([\s]+)(No|However|But|On the contrary|In contrast|Whereas)([\s]*,[\s]+)([^.!?\n]{50,}[.!?]+)([\s]+)',
        }

        self.compile_regex()
    
    def collect_mined(self, tup, class_name):
        dic = {
            'label': class_name,
            'verbalizer': tup[3],
            'sentence1': tup[1],
            'sentence2': tup[-2],
        }
        return dic

    def get_all_templates(self, entry, random_seed):
        if entry['label'] == 'Different':
            return [
            ("\"{sentence1}\" Generate a sentence that expresses a contrasting idea to the previous statement.", "{sentence2}"),
            ("Can you create a sentence that contradicts the meaning of \"{sentence1}\"?", "{sentence2}"),
            ("Given the sentence \"{sentence1}\", can you come up with a statement that contradicts its meaning?", "{sentence2}"),
            ("Here is a sentence: \"{sentence1}\". Now, provide a sentence that contradicts its meaning.", "{sentence2}"),
            ("Your challenge is to create a sentence that expresses the opposite of \"{sentence1}\". Answer:", "{sentence2}"),
            ("Contradict the meaning of the sentence \"{sentence1}\" by crafting another sentence.", "{sentence2}"),
            ("Compose a sentence that contradicts the idea conveyed in \"{sentence1}\".", "{sentence2}"),
            ("Can you generate a sentence that has a conflicting meaning compared to \"{sentence1}\"?", "{sentence2}"),
            ("In opposition to the sentence \"{sentence1}\", create a sentence with a contradictory meaning.", "{sentence2}"),
            ("Your task is to provide a sentence that negates or contradicts the message of \"{sentence1}\".", "{sentence2}"),
            ("Given the sentence \"{sentence1}\", come up with a different sentence that contradicts its meaning?", "{sentence2}"),
            ("Craft a sentence that goes against the meaning of the sentence \"{sentence1}\".", "{sentence2}"),
            ]
        elif entry['label'] == 'Similar':
            return [
            ('Complete the following sentence: {sentence1} Namely,', "{sentence2}"),
            ('{sentence1} In other words\nProvide the missing portion of the above sentence:', "{sentence2}"),
            ('Q: {sentence1} That is to say?', "{sentence2}"),
            ("Question: Complete the following statement using \"{verbalizer}\" in relation to \"{sentence1}\"\nAnswer:", "{sentence2}"),
            ("Question: {sentence1} {verbalizer}?", "{sentence2}"),
            ("{sentence1} {verbalizer},\nHow do you finish this sentence?", "{sentence2}"),
            ("Extend the thought in this sentence: {sentence1} To elaborate further:","{sentence2}"),
            ("Build upon the statement {sentence1} by utilizing \"{verbalizer}\" to express the following concept.", "{sentence2}"),
            ("\"{sentence1}\" Generate a sentence that expresses a further elaboration to the previous statement.", "{sentence2}"),
            ("\"{sentence1}\" Expand on the previous statement:", "{sentence2}"),
            ("{sentence1}\nProvide an explanatory sentence:", "{sentence2}"),
            ]
    
    def format_single_demo(self, entry, random_seed):
        kw_dic = {}
        kw_dic['verbalizer'] = entry['verbalizer']
        kw_dic['sentence1'] = entry['sentence1']
        kw_dic['sentence2'] = entry['sentence2'][0].upper() + entry['sentence2'][1:]    
        
        template = self.get_template(entry, random_seed)
        return self.fill_in_the_template(template, kw_dic)

@type_map.add("word2text")
class word2text(BaseType):
    def __init__(self):
        super().__init__()
        self.mine_regex = {
        'definition': r'([\s]+)([^.!?,;\s\"]{10,})([\s]+)(is defined as|\'s definition is)([\s]+)([^.!?\n]{20,}[.!?]+)([\s]+)',
        'topic': r'([.!?]+[\s]+)([^.!?,;\n]{20,})([\s]+)(was about|talks about|is about|\'s topic is)([\s]+)([^.!?\n]{20,}[.!?]+)([\s]+)',
	    } 
        # `topic` is defined as a summaization task in our paper, 
        # here we categorize it to word2text for simple code implementation
        
        self.compile_regex()
        
        self.min_kw_num = 3 # requires at least 3 domain-specific keywords, 
        self.max_sent_len = 100 # with fewer than 100 sent tokens.
        self.max_collect_sent = 2 # early break when find enough task examples.

    def collect_mined(self, tup, class_name):
        if class_name == 'definition':
            dic = {
                'relation': class_name,
                'verbalizer': tup[3],
                'word': tup[1],
                'definition': tup[-2],
            }
        elif class_name == 'topic':
            dic = {
                'relation': class_name,
                'verbalizer': tup[3],
                'sentence': tup[1],
                'topic': tup[-2],
            }
        return dic
    def mine(self, text, domain, sents, **kwargs):  
        def mine_regex(text):
            mined_dic = {}
            mined_num = 0
            for class_name, regex in self.regex_dic.items(): 
                mined_dic[class_name]=[]
                x = regex.findall(text)
                if len(x)>0:
                    for tup in x:
                        collected = self.collect_mined(tup, class_name)
                        mined_dic[class_name].append(collected)   
                mined_num += len(mined_dic[class_name])
            return mined_dic, mined_num
        mined_dic, mined_num = mine_regex(text)

        random.seed(len(text)) # fix random seed for reproduction
        random.shuffle(sents)

        mined_dic['word2text']=[] # wrap as a list to align with other task types
        for sent in sents:
            if len(mined_dic['word2text']) == self.max_collect_sent: 
                break
            sent_tokens = set(self.domain_spm.encode(sent, out_type=str))
            specific_tokens_in_sent = list(self.specific_token_set & sent_tokens)
            if len(specific_tokens_in_sent) >= self.min_kw_num and len(sent_tokens) <= self.max_sent_len: 
                tokens = [self.domain_spm.decode(token) for token in specific_tokens_in_sent] # transfer tokens back to normal words
                dic = {
                    'relation': 'word2text',
                    'domain': domain,
                    'token_set': tokens,
                    'sent': sent.strip(),
                }
                mined_dic['word2text'].append(dic)
        mined_num += len(mined_dic['word2text'])     
        return mined_dic, mined_num
    
    def get_all_templates(self, entry, random_seed):
        if entry['relation'] == 'word2text':
            return [
            ("Concepts: {tripleset}\nWrite a sentence that includes all these {domain} words.\nSentence:", "{target}"),
            ("Concepts: {tripleset}\nFind a sentence in the article that includes all these words in the {domain} domain.\nSentence:", "{target}"),
            ("Keywords: {tripleset}\nWhat is a sentence that includes all these {domain} keywords?", "{target}"),
            ("Here are some concepts: {tripleset}\nWhat is a sentence about these {domain} concepts in the article?", "{target}"),
            ("Produce a sentence which mentions all of these {domain} concepts: {tripleset}\nAnswer:", "{target}"),
            ("Write a {domain} sentence about the following things:\n{tripleset}\nAnswer:", "{target}"),
            ("Generate a sentence that includes all the following {domain} words: {tripleset}. Sentence:", "{target}"),
            ("Sentence: {target}\nWhat are the keywords about {domain} in this sentence?", "{tripleset}"),
            ("What are the most important words about {domain} in the following sentence\n{target}\nWords:", "{tripleset}"),
            ("{target}\nIdentify the most salient words about {domain} in the above sentence.", "{tripleset}"),
            ("Concepts: {tripleset}\nWhat would a {domain} sentence about these concepts be like?", "{target}"),
            ("Here are some words about {domain}: {tripleset}.\nWrite a sentence that describes them.", "{target}"),
            ("Here are some {domain} words: {tripleset}.\nTell me a sentence that describes them in the article.", "{target}"),
            ("Here are some concepts about {domain}: {tripleset}.\nGenerate a detailed description of them.\nDescription:", "{target}"),
            ("Generate a {domain} sentence about: {tripleset}\nSentence:", "{target}"),
            ("Write a {domain} sentence about [{tripleset}].", "{target}"),
            ("Produce a long descriptive sentence about {domain} that uses all these words: {tripleset}.\nSentence:", "{target}"),
            ("Create a set of three {domain} concepts in the following sentence.\n{target}\nConcepts:", "{tripleset}"),
            ("{tripleset}\nWhat is the sentence in the article that verbalizes these {domain} concepts?", "{target}"),
            ("Keywords: {tripleset}\nTell me the sentence in the article about these {domain} concepts.\nSentence:", "{target}"),
            ("Here are some {domain} keywords: {tripleset}.\nWrite a sentence that includes them.", "{target}"),
            ("Generate a sentence that includes these {domain} keywords [{tripleset}].", "{target}"),
            ("Find a sentence in the above article that includes the following {domain} words: [{tripleset}].", "{target}"),
            ("Produce a long descriptive {domain} sentence that uses all these words: {tripleset}\nAnswer:", "{target}"),
            ("Sentence: {target}\nWhat keywords about {domain} can be extracted from this sentence?", "{tripleset}"),
            ]
        elif entry['relation'] == 'definition':
            return [
                ("Q: {word} {verbalizer}? A:", "{definition}"),
                ("Next question: {word} {verbalizer}:", "{definition}"),
                ("{word} {verbalizer}?", "{definition}"),
                ("{word} {verbalizer}:", "{definition}"),
                ("What is the definition of {word}?", "{definition}"),
                ("How to define {word}?", "{definition}"),
                ("Explain the meaning of \"{word}\".", "{definition}"),
                ("What does \"{word}\" refer to?", "{definition}"),
                ("Please elucidate the concept of {word}\nAnswer:", "{definition}"),
                ("What is the meaning of the term \"{word}\"?", "{definition}"),
                ("Could you offer a definition for {word}?", "{definition}"),
                ("Could you offer a definition for {word}?\nDefinition:", "{definition}"),
            ]
        elif entry['relation'] == 'topic':
            return [
                ("{sentence} {verbalizer}?", "{topic}"),
                ("{sentence} {verbalizer}:", "{topic}"),
                ("Q: {sentence} {verbalizer}? A:", "{topic}"),
                ("Answer the question\n{sentence} {verbalizer}?", "{topic}"),
                ("Answer the question\n{sentence} {verbalizer}:", "{topic}"),        
                ("Answer the following question:\n{sentence} {verbalizer}?\nAnswer:", "{topic}"),
                ("Answer this question:\n{sentence} {verbalizer}?", "{topic}"),
                ("Please answer this question: {sentence} {verbalizer}?\nAnswer:", "{topic}"),
                ("Answer the question...{sentence} {verbalizer}?", "{topic}"),
                ("Can you tell me the answer to \"{sentence} {verbalizer}?\"?", "{topic}"),
                ("Next question: {sentence} {verbalizer}:", "{topic}"),
                ("Q: {sentence} {verbalizer}:", "{topic}"),
                ("Please answer this question: {sentence} {verbalizer}:", "{topic}"),
                ("Write the answer: {sentence} {verbalizer}?\nAnswer:", "{topic}"),
                ("What is the answer to \"{sentence} {verbalizer}:\"?", "{topic}"),
                ("Answer this question.\n{sentence} {verbalizer}:", "{topic}"),
                ("Answer the following question. {sentence} {verbalizer}:", "{topic}"),
                ("Question: {sentence} {verbalizer}?", "{topic}"),
                ("{sentence} {verbalizer}??", "{topic}"),
            ]

    def format_single_demo(self, entry, random_seed):
        kw_dic = {}
        if entry['relation'] == 'word2text':
            kw_dic['domain'] = entry['domain']
            kw_dic['tokens'] = entry['token_set']
            kw_dic['tripleset'] = ', '.join(kw_dic['tokens'][:self.min_kw_num])
            kw_dic['target'] = entry['sent'].strip()
        elif entry['relation'] == 'definition' or entry['relation'] == 'topic':
            kw_dic = entry

        template = self.get_template(entry, random_seed)
        return self.fill_in_the_template(template, kw_dic)

@type_map.add("summarize")
class summarize(BaseType):
    def __init__(self):
        super().__init__()

    def mine(self, text, domain, title, **kwargs):
        # seems redundant but has to do so to align with other task types
        mined_dic={'title':title, 'domain': domain}
        mined_num = 1 if title is not None else 0
        return mined_dic, mined_num
    
    def get_all_templates(self, entry, random_seed):
        # those are templates when summarization is conducted but text completion is NOT conducted
        return [
            #summary_templates
            ("{context_wo_title}\n\nWhat is a potential title for this context in the {domain} domain?\nTitle: {title}{qa_demos}"),
            ("{domain} article: {context_wo_title}{qa_demos}\n\nWhat is the title of this article? {title}"),
            ("Article: {context_wo_title}{qa_demos}\n\nGenerate a title for this {domain} paragraph.\nTitle: {title}"),
            ("{context_wo_title}\n\nWrite a title for the above {domain} article. {title}{qa_demos}"),
            ("{context_wo_title}\nBriefly summarize this {domain} text? {title}{qa_demos}"),
            ("Article in the {domain} domain: {context_wo_title}\n\nGenerate a short summary for this article.\nAnswer: {title}{qa_demos}"),
            ("{context_wo_title}{qa_demos}\n\nSummarize the aforementioned {domain} text in a single sentence. {title}"),
            ("{context_wo_title}\nCan you generate a short summary of the above {domain} paragraph? {title}{qa_demos}"),
            ("{context_wo_title}\nPlease write a short summary for the above article in the {domain} domain. {title}{qa_demos}"),
            ("Context: {context_wo_title}{qa_demos}\n\nWhat was this {domain} article about? {title}"),
            # write based on title
            ("Write an article about {domain} domain, using the following title: {title}.\nArticle: {context_wo_title}{qa_demos}"),
            ("Title: {title}\nWrite a an article about {domain} domain based on this title. {context_wo_title}{qa_demos}"),
            ("Use the title \"{title}\" to write a {domain} article.\nArticle: {context_wo_title}{qa_demos}"),
            ("Craft an informative article about the {domain} domain, drawing from the following summary: {title}\nArticle: {context_wo_title}{qa_demos}"),
            ("Create a {domain} article inspired by the provided title: {title}\nOutput: {context_wo_title}{qa_demos}"),
            ("Can you develop an engaging {domain} article using the title \"{title}\"? {context_wo_title}{qa_demos}"),
            ("Write an informative piece on the {domain} domain, using the provided title: {title}. {context_wo_title}{qa_demos}"),
            ("Craft an article focused on {domain}, utilizing the provided title: {title}.\nArticle: {context_wo_title}{qa_demos}"),
            ("Compose an in-depth {domain} article based on the title: {title}\nArticle: {context_wo_title}{qa_demos}"),
            ("Can you create an article delving into the {domain} domain, incorporating the given title \"{title}\"? {context_wo_title}{qa_demos}"),
        ]
    
    def format_single_demo(self, entry, random_seed):
        sents = entry.pop('sents')
        template = self.get_template(entry, random_seed)
        
        entry['context_wo_title'] = ''.join(sents).strip()
        final_demo = template.format(**entry)
        return final_demo

@type_map.add("text_completion")
class text_completion(BaseType):
    def __init__(self):
        super().__init__()

    def mine(self, domain, sents, **kwargs):
        # seems redundant but has to do so to align with other task types
        mined_dic={'sents':sents, 'domain': domain}  
        mined_num = 1 if len(sents) >= 4 else 0
        return mined_dic, mined_num
    
    def get_all_templates(self, entry, random_seed):
        # those are templates when text completion is conducted but summarization is NOT conducted
        return [
            ("Please complete an article about {domain}: {context_1st_half} {context_2nd_half}{qa_demos}"),
            ("Here is the first part of an article about {domain}: {context_1st_half}\n\nHow would you continue the article? {context_2nd_half}{qa_demos}"),
            ("Explore the initial section of an article on {domain}: {context_1st_half}\nWhat could be the next part? {context_2nd_half}{qa_demos}"),
            ("Read the beginning of an article about {domain}: {context_1st_half}\n\nWrite the subsequent part? {context_2nd_half}{qa_demos}"),
            ("In this article snippet about {domain}, you will find the first part: {context_1st_half}\nHow would you compose the remaining section? {context_2nd_half}{qa_demos}"),
            ("Take a look at the introductory part of an article on {domain}: {context_1st_half}\n\nYour challenge is to write the following segment\nAnswer: {context_2nd_half}{qa_demos}"),
            ("Review the initial portion of an article discussing {domain}: {context_1st_half}\nWhat would you include in the rest of the article? {context_2nd_half}{qa_demos}"),
            ("Consider the first segment of an article centered around {domain}: {context_1st_half}\nContinuation of the article: {context_2nd_half}{qa_demos}"),
            ("Examine the first segment of an article exploring {domain}: {context_1st_half}\n\nQuestion: Complete the article?\nCompletion: {context_2nd_half}{qa_demos}"),
            ("Read the beginning of an article on {domain}: {context_1st_half}{qa_demos}\n\nHow would you extend the article? {context_2nd_half}"),
        ]
    
    def format_single_demo(self, entry, random_seed):
        sents = entry.pop('sents')
        entry['context_1st_half'] = entry['title']+'\n' if entry['title'] is not None else ''

        cut_index = random.Random(random_seed).randint(1, len(sents)-1)

        entry['context_1st_half'] += ''.join(sents[:cut_index]).strip()
        entry['context_2nd_half'] = ''.join(sents[cut_index:]).strip()
        template = self.get_template(entry, random_seed)
        final_demo = template.format(**entry)
        return final_demo


@type_map.add("summarize_completion")
class summarize_completion(BaseType):
    def __init__(self):
        super().__init__()

    def get_all_templates(self, entry, random_seed):
        # applicable to both text completion and summarization:
        return [
            ("Please complete an article about {domain}: {context_1st_half} {context_2nd_half}{qa_demos}\n\nWhat was this article about?\nAnswer: {title}"),
            ("Here is the first part of an article about {domain}: {context_1st_half}\n\nPlease complete it.\nCompletion: {context_2nd_half}{qa_demos}\n\nWhat was this article about? {title}"),
            ("Explore the initial section of an article on {domain}: {context_1st_half}\n\nProvide the text ending? {context_2nd_half}\n\nPropose a title for this context? {title}{qa_demos}"),
            ("Read the beginning of an article about {domain}: {context_1st_half}\n\nYour task is to add the subsequent part. {context_2nd_half}\n\nBriefly summarize this text. Summary: {title}{qa_demos}"),
            ("In this article snippet about {domain}, you will find the first half: {context_1st_half}\n\nCompose the remaining section: {context_2nd_half}\n\nWrite a title for it.\nTitle: {title}{qa_demos}"),
            ("Take a look at the first part of an article on {domain}: {context_1st_half}\n\nYour challenge is to write the following segment. {context_2nd_half}\n\nWhat is a very short summary of the above text? {title}{qa_demos}"),
            ("Review the initial portion of an article discussing {domain}: {context_1st_half}\n\nWhat would you include in the rest of the article? {context_2nd_half}\n\nWhat is a shorter version of this article?\nShort version: {title}{qa_demos}"),
            ("Consider the opening of an article centered around {domain}: {context_1st_half}\n\nNow, provide the continuation of the article.\nContinuation: {context_2nd_half}\n\nWhat was this article about? {title}{qa_demos}"),
            ("Examine the first segment of an article exploring {domain}: {context_1st_half}\n\nComplete the article? {context_2nd_half}\nCan you generate a short summary of the above paragraph?\nAnswer: {title}{qa_demos}"),
            ("Read the beginning of an article on {domain}: {context_1st_half}\n\nHow would you extend the article? {context_2nd_half}\n\nPlease write a short summary for the above article. {title}{qa_demos}")
        ]
    
    def format_single_demo(self, entry, random_seed):
        sents = entry.pop('sents')
        template = self.get_template(entry, random_seed)
        cut_index = random.Random(random_seed).randint(1, len(sents)-1)

        entry['context_1st_half'] = ''.join(sents[:cut_index]).strip()
        entry['context_2nd_half'] = ''.join(sents[cut_index:]).strip()
        final_demo = template.format(**entry)
        return final_demo

@type_map.add("no_summarize_completion")
class no_summarize_completion(BaseType):
    def __init__(self):
        super().__init__()

    def get_all_templates(self, entry, random_seed):
        # applicable to having no summarization and no completion
        return [
            ("Please answer some questions about the following article in the {domain} domain:\n{context}{qa_demos}"),
            ("Read this {domain} article and answer questions\n{context}{qa_demos}"),
            ("{context}{qa_demos}"),
            ("Answer some questions about this article about {domain} domain:\n{context}{qa_demos}"),
            ("Here are some questions about this {domain} article: {context}{qa_demos}"),
            ("{domain} article: {context}{qa_demos}"),
            ("Article: {context}{qa_demos}"),
            ("Read this article in the {domain} domain: {context}{qa_demos}"),
            ("Given the {domain} article: {context}{qa_demos}"),
            ("Context: {context}{qa_demos}"),
            ("Article in the {domain} domain: {context}{qa_demos}"),
            ("Use this {domain} article to answer the questions: {context}{qa_demos}"),
            ("Answer based on context in the {domain} domain:\n{context}{qa_demos}"),
        ]
    
    def format_single_demo(self, entry, random_seed):
        sents = entry.pop('sents')
        entry['context'] = entry['title']+'\n' if entry['title'] is not None else ''
       
        template = self.get_template(entry, random_seed)

        entry['context'] += ''.join(sents).strip()
        final_demo = template.format(**entry)
        return final_demo

    
@type_map.add("overall")
class overall(BaseType):
    def __init__(self):
        super().__init__()
        self.demo_deliminator='\n\n'
        self.intro_deliminators=[ # connect raw text with the followed QAs
            ("\nPlease answer some questions about the above article in the {domain} domain:\n\n"),
            ("\n\n"),
            ("\nAnswer some questions about the above article about {domain} domain:\n\n"),
            ("\n\nWhat are the answers to these questions?\n"),
            ("\n\nNow answer these questions:\n\n"),
            ("\nNow answer the following questions:\n\n"),
            ("\n\nWhat are the answers to the questions or completions:\n"),
            ("\nHow would one answer these questions in the {domain} domain:\n\n"),
            ("\n\nUse evidence from the {domain} article to answer these questions:\n\n"),
            ("\n\nUse this above {domain} article to answer the questions:\n"),
            ("\nAnswer the following {domain} questions based on the article:\n\n"),
            ("\n\n"),
            ("\nAnswer these {domain} questions:\n"),
            ("\n\nBased on the above article in the {domain} domain, answer questions.\n\n"),
            ("\nWrite some question-answer pairs about the above {domain} article:\n\n"),
            ('\nRespond to the following questions based on the above article:\n\n'),
            ('\n\nUpon reading the article, answer the following questions:\n\n'),
            ('\nEvaluate your understanding of the article by answering the following questions:\n\n')
        ]
    
    def format_recomprehension(self, overall_entry, insert_types=TYPES):
        qa_demo_list=[]
        seed = overall_entry['text_id']
        count_dict={}
        for type in list(set(insert_types) & set(['nli', 'common_reason', 'paraphrase', 'word2text'])):
            type_cls = type_map.cls_dic[type]()
            type_examples = []
            count_dict[type]={}
            for subcategory, examples in overall_entry[type].items():
                if len(examples) == 0: continue
                random.Random(seed).shuffle(examples)
                type_examples += examples[:type_cls.max_subcategory_num]
                count_dict[type][subcategory] = len(examples[:type_cls.max_subcategory_num])
            if len(type_examples) == 0: continue
            # ensure examples of one type altogether, to imitate the few-shot setting
            demo = self.demo_deliminator.join([type_cls.format_single_demo(example, seed) for example in type_examples]) 
            qa_demo_list.append(demo)

        if len(qa_demo_list) > 0:
            random.Random(seed).shuffle(qa_demo_list)
            intro_template = random.Random(seed).choice(self.intro_deliminators)
            intro = intro_template.replace('{domain}', overall_entry['summarize']['domain'])
            qa_demos = f'{intro}{self.demo_deliminator.join(qa_demo_list)}'
        else:
            qa_demos = ''

        def summaize_only(count_dict):
            count_dict['summarize'] = 1
            count_dict['text_completion'] = 0
            overall_cls = summarize()
            entry = overall_entry['summarize']
            entry['sents'] = overall_entry['text_completion']['sents']
            entry['qa_demos'] = qa_demos
            entry['spm'] = self.ori_spm
            read_compre_demo = overall_cls.format_single_demo(entry, seed)
            return remove_double_space(read_compre_demo), count_dict
        
        def completion_only(count_dict):
            count_dict['summarize'] = 0
            count_dict['text_completion'] = 1
            overall_cls = text_completion()
            entry = overall_entry['text_completion']
            entry['qa_demos'] = qa_demos
            entry['title'] = overall_entry['summarize']['title']
            entry['spm'] = self.ori_spm
            read_compre_demo = overall_cls.format_single_demo(entry, seed)
            return remove_double_space(read_compre_demo), count_dict
        
        def summarize_and_completion(count_dict):
            count_dict['summarize'] = 1
            count_dict['text_completion'] = 1
            overall_cls = summarize_completion()
            entry = overall_entry['text_completion']
            entry['qa_demos'] = qa_demos
            entry['title'] = overall_entry['summarize']['title']
            entry['spm'] = self.ori_spm
            read_compre_demo = overall_cls.format_single_demo(entry, seed)
            return remove_double_space(read_compre_demo), count_dict
        
        def no_summarize_or_completion(count_dict):
            count_dict['summarize'] = 0
            count_dict['text_completion'] = 0
            overall_cls = no_summarize_completion()
            entry = overall_entry['text_completion']
            entry['qa_demos'] = qa_demos
            entry['title'] = overall_entry['summarize']['title']
            entry['spm'] = self.ori_spm
            read_compre_demo = overall_cls.format_single_demo(entry, seed)
            return remove_double_space(read_compre_demo), count_dict
        
        if ('summarize' in insert_types and overall_entry['summarize']['title'] is not None) and ('text_completion' in insert_types and len(overall_entry['text_completion']['sents']) >=2):
            np.random.seed(seed)
            read_func = np.random.choice([summaize_only, completion_only, summarize_and_completion, no_summarize_or_completion], p=[0.4, 0.1, 0.4, 0.1])
        elif ('summarize' in insert_types and overall_entry['summarize']['title'] is not None):
            np.random.seed(seed)
            read_func = np.random.choice([summaize_only, no_summarize_or_completion], p=[0.5, 0.5])
        elif ('text_completion' in insert_types and len(overall_entry['text_completion']['sents']) >=2):
            np.random.seed(seed)
            read_func = np.random.choice([completion_only, no_summarize_or_completion], p=[0.5, 0.5])
        else:
            read_func = no_summarize_or_completion
        
        return read_func(count_dict)

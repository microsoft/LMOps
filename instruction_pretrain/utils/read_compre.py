import random
import sys
sys.path.append("./")
import utils.read_compre_pt as rc_pt_utils
from tqdm import tqdm

read_template = ' <CON> {context} </CON>' # please do NOT strip the beginning space
QA_template = '<QUE> {question} <ANS> {answer} </END>'
delimiter = '\n\n'
bos_token = '<s>'
eos_token = '</s>'

def cook_read(raw_context, bos_token='<s>'):
    """Format the reading material"""
    return bos_token + read_template.replace('{context}', raw_context) + delimiter if len(raw_context) > 0 else bos_token


def cook_compre(QA_list):
    """Format comprehension tasks"""
    assert len(QA_list) > 0, f'len(QA_list) == {len(QA_list)}'
    comprehension_list = []
    for qa_entry in QA_list:
        qa = QA_template.replace('{question}', qa_entry['Q']).replace('{answer}', qa_entry['A'])
        comprehension_list.append(qa)
    return delimiter.join(comprehension_list)


def cook_one_entry(entry, bos_token='<s>'):
    """Format the reading material and comprehension tasks"""
    return cook_read(entry['context'], bos_token), cook_compre(entry['QA_list'])


def deduplicate_questions(QA_list):
    deduplicated_list = []
    raw_questions = []
    for qa_entry in QA_list:
        if qa_entry['Q'].lower() not in raw_questions:
            deduplicated_list.append(qa_entry)
            raw_questions.append(qa_entry['Q'].lower())
    return deduplicated_list


def parse_comprehension(compre_str, deduplicate = False):
    """Get the list of QAs from the generated comprehension text"""
    QA_str_list = compre_str.split('</END>') # the list of `\n\n<QUE> {question} <ANS> {answer} `
    if not compre_str.endswith('</END>'):
        # This menas the last QA pair is incomplete, remove it
        QA_str_list = QA_str_list[:-1]

    QA_list = []
    for QA_str in QA_str_list:
        try:
            assert len(QA_str.split('<ANS>')) == 2, f'invalid QA string: {QA_str}'
            Q_str, A_str = QA_str.split('<ANS>') # `\n\n<QUE> {question} `, ` {answer}`
            Q_str, A_str = Q_str.strip(), A_str.strip() # `<QUE> {question}`, `{answer}`
            assert Q_str.startswith('<QUE>'), f'invalid question string: {Q_str} in QA_str: {QA_str}'
            assert len(A_str)>0, f'invalid answer string in QA_str: {QA_str}'
            Q_str = Q_str.replace('<QUE>', '').strip()
            QA_list.append({'Q': Q_str, 'A': A_str})
        except Exception as e:
            continue

    if deduplicate:
        # deduplicate questions
        QA_list = deduplicate_questions(QA_list)

    return QA_list


def cook_pt_entries(read_collection, random_seed):
    # option related
    opt_start = random.Random(random_seed).choice(rc_pt_utils.OPT_START_STRING_CANDIDATES)
    opt_item_name = random.Random(random_seed).choice(rc_pt_utils.OPT_ITEM_NAME_CANDIDATES)
    opt_item_end = random.Random(random_seed).choice(rc_pt_utils.OPT_ITEM_END_STR_CANDIDATES)
    opt_delimiter = random.Random(random_seed).choice(rc_pt_utils.OPT_DELIMITER_CANDIDATES)
    unanswerable_option = random.Random(random_seed).choice(rc_pt_utils.UNANSWERABLE_OPTIONS)

    # delimiter between rc texts
    rc_demo_delimiter = random.Random(random_seed + 1).choice(rc_pt_utils.FEWSHOT_PATTERNS['qa']).example_separator

    pattern_dict = {}
    # the following fs patterns might be changed according to rc_mode, 
    # so we put them in the pattern_dict, which might be updated in `get_patterns()` function
    pattern_dict['fs_basic_pattern'] = random.Random(random_seed).choice(rc_pt_utils.FEWSHOT_PATTERNS['qa'])
    pattern_dict['fs_option_pattern'] = random.Random(random_seed).choice(rc_pt_utils.FEWSHOT_PATTERNS['qa_w_option'])
    pattern_dict['fs_basic_cot_pattern'] =random.Random(random_seed).choice(rc_pt_utils.FEWSHOT_PATTERNS["qa_w_cot"])
    pattern_dict['fs_option_cot_pattern'] = random.Random(random_seed).choice(rc_pt_utils.FEWSHOT_PATTERNS["qa_w_option_w_cot"])
    
    # retrieve rc_context from the last rc entry
    res_list = []
    idx = len(read_collection) - 1
    rc_entries_in_one_text = []
    while idx >= 0:
        read_entry = read_collection[idx]
        read_entry['QA_list'] = rc_pt_utils.parse_QA_list(read_entry['QA_list'], opt_start, opt_item_name, opt_item_end, opt_delimiter, unanswerable_option, random_seed=random_seed)
        # the loop starts from the end, so we prepend the new collected entry to rc_entries_in_one_text
        rc_entries_in_one_text = [read_entry] + rc_entries_in_one_text

        assert 'inherit_from_prev' in read_entry

        if not read_entry['inherit_from_prev']:
            # prev rc does NOT connect with the current one
            # process current rc_entries_in_one_text to one piece of fewshot_pt_rc
            # and reset rc_entries_in_one_text
            min_num_of_QAs = min(len(rc_entry['QA_list']) for rc_entry in rc_entries_in_one_text) # to decide rc_mode
            pattern_dict = rc_pt_utils.get_patterns(pattern_dict, random_seed, min_num_of_QAs)
            pt_rc_list = [] # save all the rc for formating one few-shot-rc
            for read_entry in rc_entries_in_one_text:
                if len(read_entry['QA_list']) == 0:
                    res_list = [read_entry['context']] + res_list
                    continue
                one_pt_rc = rc_pt_utils.format_one_pt_rc(read_entry, pattern_dict)
                pt_rc_list.append(one_pt_rc)
            fewshot_pt_rc = rc_demo_delimiter.join(pt_rc_list) # get one piece of fewshot_pt_rc
            res_list = [fewshot_pt_rc] + res_list
            
            rc_entries_in_one_text = [] # reset rc_entries_in_one_text
            
        idx -= 1
    
    assert len(rc_entries_in_one_text) == 0, f'len(rc_entries_in_one_text): {len(rc_entries_in_one_text)}!=0\nrc_entries_in_one_text:\n{rc_entries_in_one_text}'
    return res_list


def process_syn_raw_text(data):
    """Process the output sequence from the instruction synthesizer"""
    read_collection_list = []
    for entry in data:
        cur_QA_list = parse_comprehension(entry['pred'], deduplicate = True)
        prev_read_collection = entry.get('prev_read_collection', [])
        # , 'pred': entry['pred']
        cur_entry = {'context': entry['text'], 'QA_list': cur_QA_list}

        if len(prev_read_collection) > 0:
            # check if the cur_entry successfully inherit from the latest prev_read_entry
            prev_entry = prev_read_collection[-1]

            if prev_entry['followed_by_next']:
                # followd_by_next ONLY indicates the prev_entry is prepended the cur_context when synthesizing the cur_entry
                # we need the 'inherit_from_prev' to indicates whether cur_entry indeed inherits from the prev_entry
                cur_entry['inherit_from_prev'] = True if len(cur_QA_list) > 0 else False
            else:
                # prev_entry is not presented when synthesizing the cur_entry
                cur_entry['inherit_from_prev'] = False
        else:
            cur_entry['inherit_from_prev'] = False

        # save the cur_entry to the 'prev_read_collection'
        prev_read_collection.append(cur_entry)
        read_collection_list.append(prev_read_collection)

    return read_collection_list


def get_dataset(prev_examples, cur_raw_texts, max_model_len, max_new_tokens):
    cur_raw_data = [{'text': text} for text in cur_raw_texts]
    for id, cur_raw_entry in tqdm(enumerate(cur_raw_data), disable=False):
        cur_raw_entry['id'] = id # assgin index
        if id >= len(prev_examples):
            # NO available previous read_compre data
            cur_raw_entry['prev_read_collection'] = []
            cur_raw_entry['cur_context'] = cook_read(cur_raw_entry['text'])
            continue

        prev_read_collection = prev_examples[id]
        
        # acculumative previous context
        if len(prev_read_collection[-1]['QA_list']) > 0:
            prev_context = cook_read(prev_read_collection[-1]['context']) + cook_compre(prev_read_collection[-1]['QA_list']) + eos_token 
            idx = len(prev_read_collection) - 1
            # retrieve more previous contexts
            while idx - 1 >= 0:
                if prev_read_collection[idx]['inherit_from_prev']:
                    # prepend the context of 'the entry previous to the current one'
                    idx -= 1
                    prev_read_entry = prev_read_collection[idx]
                    prev_context = cook_read(prev_read_entry['context']) + cook_compre(prev_read_entry['QA_list']) + eos_token + prev_context  
                else:
                    # failed to retrieve previous entries
                    break
        else:
            prev_context = ''

        # check if the acculmuated previous context + cur_context < self.input_length
        if len(prev_context) > 0:
            tmp_cur_context = prev_context  + cook_read(cur_raw_entry['text'])
            # tmp_len = len(tokenizer.tokenize(tmp_cur_context))
            tmp_len = len(tmp_cur_context.split(" "))*1.5 # NOTE use tokenizer for calculating length is too slow, we roughly count it

            if tmp_len <= max_model_len - max_new_tokens:
                prev_read_collection[-1]['followed_by_next'] = True
                cur_context = tmp_cur_context
            else:
                # fail to follow latest entry
                prev_read_collection[-1]['followed_by_next'] = False
                cur_context = cook_read(cur_raw_entry['text'])
        else:
            prev_read_collection[-1]['followed_by_next'] = False
            cur_context = cook_read(cur_raw_entry['text'])

        cur_raw_entry['prev_read_collection'] = prev_read_collection
        cur_raw_entry['cur_context'] = cur_context

    if len(prev_examples) > len(cur_raw_data):
        # the above for-loop assigns prev_examples[:len(cur_raw_data)] to cur_raw_data
        # to avoid prev_examples[len(cur_raw_data):] being dumped in this turn, we prepend them to some entries in cur_raw_data
        # and set the followed_by = False, so that those data would NOT be connected to the cur_raw_data in pre-train
        individual_prev_examples = prev_examples[len(cur_raw_data):]
        for idx, prev_read_collection in enumerate(individual_prev_examples):
            assert 'followed_by_next' not in prev_read_collection[-1]
            prev_read_collection[-1]['followed_by_next'] = False
            cur_raw_data[idx]['prev_read_collection'] = prev_read_collection + cur_raw_data[idx]['prev_read_collection']

    return cur_raw_data


def run(split, llm, sampling_params):
    prompts = [entry['cur_context'] for entry in split]
    try:
        outputs = llm.generate(prompts, sampling_params)
    except Exception as e:
        print(f'raise error: {e}, skip and return nothing')
        outputs = ['DUMMY_RESPONSE']*len(prompts)
    metadata_list = []
    for id, output in enumerate(outputs):
        metadata = split[id]
        metadata.update({'pred': output.outputs[0].text})
        if output.prompt != metadata['cur_context']:
            print(f'output.prompt:{output.prompt} != metadata["cur_context"]: {metadata["cur_context"]}')
        metadata_list.append(metadata)
    return process_syn_raw_text(metadata_list)
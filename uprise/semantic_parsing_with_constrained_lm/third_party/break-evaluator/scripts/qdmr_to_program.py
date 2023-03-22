from pathlib import Path
import os
import argparse
import traceback
import pandas as pd

import re
from enum import Enum

DELIMITER = ';'
REF = '#'

pd.set_option('display.max_colwidth', -1)

def parse_decomposition(qdmr):
    """Parses the decomposition into an ordered list of steps

 Parameters
 ----------
 qdmr : str
     String representation of the QDMR

 Returns
 -------
 list
     returns ordered list of qdmr steps
 """
    crude_steps = qdmr.split(DELIMITER)
    steps = []
    for i in range(len(crude_steps)):
        step = crude_steps[i]
        tokens = step.split()
        step = ""
        # remove 'return' prefix
        for tok in tokens[1:]:
            step += tok.strip() + " "
        step = step.strip()
        steps += [step]
    return steps


def extract_position_relations(qdmr_step):
    """Extract a relation regarding entity positions
     in a QDMR step. Relevant for VQA data

 Parameters
 ----------
 qdmr_step : str
     string of the QDMR step containg relative position knowledge.
     Either a FILTER of BOOLEAN step.

 Returns
 -------
 str
     string of the positional relation.
 """
    if ' left ' in qdmr_step:
        return 'POS_LEFT_OF'
    elif ' right ' in qdmr_step:
        return 'POS_RIGHT_OF'
    elif (' between ' in qdmr_step) or (' middle of ' in qdmr_step):
        return 'POS_BETWEEN'
    elif (' behind ' in qdmr_step) or (' rear of ' in qdmr_step):
        return 'POS_BEHIND_OF'
    elif (' in ' in qdmr_step and ' front ' in qdmr_step) or \
            (' infront ' in qdmr_step):
        return 'POS_IN_FRONT_OF'
    elif ' touch' in qdmr_step:
        return 'POS_TOUCHING'
    elif ' reflect' in qdmr_step:
        return 'POS_REFLECTING'
    elif (' cover' in qdmr_step) or (' obscur' in qdmr_step) or \
            (' blocking' in qdmr_step) or (' blocked' in qdmr_step) or \
            (' hidden' in qdmr_step) or (' obstruct' in qdmr_step):
        return 'POS_COVERS'
    elif (' near' in qdmr_step) or (' close ' in qdmr_step) or \
            (' closer ' in qdmr_step) or (' closest ' in qdmr_step) or \
            (' next to ' in qdmr_step) or (' adjacent ' in qdmr_step):
        return 'POS_NEAR'
    else:
        return None
    return None


### Code for QDMR step identifier:

class QDMROperation(Enum):
    FIND, SELECT, FILTER, PROJECT, AGGREGATE, GROUP, SUPERLATIVE, COMPARATIVE, UNION, \
    INTERSECTION, DISCARD, SORT, BOOLEAN, ARITHMETIC, COMPARISON, NONE = range(16)


def op_name(qdmr_op):
    return {
        QDMROperation.FIND: 'FIND',
        QDMROperation.SELECT: 'SELECT',
        QDMROperation.FILTER: 'FILTER',
        QDMROperation.PROJECT: 'PROJECT',
        QDMROperation.AGGREGATE: 'AGGREGATE',
        QDMROperation.GROUP: 'GROUP',
        QDMROperation.SUPERLATIVE: 'SUPERLATIVE',
        QDMROperation.COMPARATIVE: 'COMPARATIVE',
        QDMROperation.UNION: 'UNION',
        QDMROperation.INTERSECTION: 'INTERSECTION',
        QDMROperation.DISCARD: 'DISCARD',
        QDMROperation.SORT: 'SORT',
        QDMROperation.BOOLEAN: 'BOOLEAN',
        QDMROperation.ARITHMETIC: 'ARITHMETIC',
        QDMROperation.COMPARISON: 'COMPARISON',
        QDMROperation.NONE: 'NONE'
    }.get(qdmr_op, QDMROperation.NONE)


def step_type(step, is_high_level):
    """
 Maps a single QDMR step into relevant its operator type

 Parameters
 ----------
 step : str
     String representation a single QDMR step
 is_high_level : bool
     Flag whether or not we include the high level FIND steps,
     associated with RC datasets

 Returns
 -------
 QDMROperation
     returns the type of QDMR operation of the step
 """
    step = step.lower()
    references = extract_references(step)
    if len(references) == 0:
        # SELECT step - no references to previous steps
        return QDMROperation.SELECT
    # Discrete QDMR step types:
    if len(references) == 1:
        # AGGREGATION step - aggregation applied to one reference
        aggregators = ['number of', 'highest', 'largest', 'lowest', 'smallest', 'maximum', 'minimum', \
                       'max', 'min', 'sum', 'total', 'average', 'avg', 'mean ']
        for aggr in aggregators:
            aggr_ref = aggr + ' #'
            aggr_of_ref = aggr + ' of #'
            if (aggr_ref in step) or (aggr_of_ref in step):
                return QDMROperation.AGGREGATE
    if 'for each' in step:
        # GROUP step - contains term 'for each'
        return QDMROperation.GROUP
    if len(references) >= 2 and len(references) <= 3 and ('where' in step):
        # COMPARATIVE step - '#1 where #2 is at most three'
        comparatives = ['same as', 'higher than', 'larger than', 'smaller than', 'lower than', \
                        'more', 'less', 'at least', 'at most', 'equal', 'is', 'are', 'was', 'contain', \
                        'include', 'has', 'have', 'end with', 'start with', 'ends with', \
                        'starts with', 'begin']
        for comp in comparatives:
            if comp in step:
                return QDMROperation.COMPARATIVE
    if step.startswith('#') and ('where' in step) and len(references) == 2:
        # SUPERLATIVE step - '#1 where #2 is highest/lowest'
        superlatives = ['highest', 'largest', 'most', 'smallest', 'lowest', 'smallest', 'least', \
                        'longest', 'shortest', 'biggest']
        for s in superlatives:
            if s in step:
                return QDMROperation.SUPERLATIVE
    if len(references) > 1:
        # UNION step - '#1, #2, #3, #4' / '#1 or #2' / '#1 and #2'
        is_union = re.search("^[#\s]+[and0-9#or,\s]+$", step) or \
                   re.search("^both[#\s]+[and0-9#,\s]+$",step)
        if is_union:
            return QDMROperation.UNION
    if len(references) > 1 and ('both' in step) and ('and' in step):
        # INTERSECTION step - 'both #1 and #2'
        return QDMROperation.INTERSECTION
    if (len(references) >= 1) and (len(references) <= 2) and \
            (re.search("^[#]+[0-9]+[\s]+", step) or re.search("[#]+[0-9]+$", step)) and \
            ('besides' in step or 'not in' in step):
        # DISCARD step - '#2 besides X'
        return QDMROperation.DISCARD
    if ('sorted by' in step) or ('order by' in step) or ('ordered by' in step):
        # SORT step - '#1 ordered/sorted by #2'
        return QDMROperation.SORT
    if step.lower().startswith('if ') or step.lower().startswith('is ') or \
            step.lower().startswith('are '):
        # BOOLEAN step - starts with either 'if', 'is' or 'are'
        return QDMROperation.BOOLEAN
    if step.lower().startswith('which') and len(references) > 1:
        # COMPARISON step - 'which is better A or B or C'
        return QDMROperation.COMPARISON
    if len(references) >= 1 and ('and' in step or ',' in step):
        # ARITHMETIC step - starts with arithmetic operation
        arithmetics = ['sum', 'difference', 'multiplication', 'division']
        for a in arithmetics:
            if step.startswith(a) or step.startswith('the ' + a):
                return QDMROperation.ARITHMETIC
    # Non-discrete QDMR step types:
    if len(references) == 1 and re.search("[\s]+[#]+[0-9\s]+", step):
        # PROJECT step - 'property of #2'
        return QDMROperation.PROJECT
    if len(references) == 1 and step.startswith("#"):
        # FILTER step - '#2 [condition]'
        return QDMROperation.FILTER
    if len(references) > 1 and step.startswith("#"):
        # FILTER step - '#2 [relation] #3'
        if extract_position_relations(step) != None:
            # check if relation is a valid positional relation
            return QDMROperation.FILTER
    if is_high_level:
        return QDMROperation.FIND
    return QDMROperation.NONE


def extract_references(step):
    """Extracts list of references to previous steps

 Parameters
 ----------
 step : str
     String representation of a QDMR step

 Returns
 -------
 list
     returns list of ints of referenced steps
 """
    # make sure decomposition does not contain a mere '# ' rather than reference.
    step = step.replace("# ", "hashtag ")
    # replace ',' with ' or'
    step = step.replace(", ", " or ")
    references = []
    l = step.split(REF)
    for chunk in l[1:]:
        if len(chunk) > 1:
            ref = chunk.split()[0]
            ref = int(ref)
            references += [ref]
        if len(chunk) == 1:
            ref = int(chunk)
            references += [ref]
    return references


# %%
#
# qdmr = "return professionals ;return cities of  #1 ;return #1 where #2 contain substring 'West ;return roles of  #3 ;return streets of  #3 ;return cities of  #3 ;return states of  #3 ;return #4 ,  #5 ,   #6 , #7"
# steps = parse_decomposition(qdmr)
# print(steps)
# for step in steps:
#     print(step)
#     op_num = step_type(step, True)
#     print(op_name(op_num))


# %%

class ExecQDMR:
    """
 Class representing executable QDMR operation.
 """

    def __init__(self, op_type, op_string, prev_refs_code):
        """Creaing a new executable QDMR operation

  Parameters
  ----------
  op_type : QDMROperation
      relevant operation type
  op_string : str
      string of the QDMR operation
  prev_refs_code : dict
      dictionary where each key has the code of the referneced steps
      of the current QDMR operation
  """
        self.op_type = op_type
        self.op_string = op_string
        self.prev_refs_code = prev_refs_code
        self.arguments = self.get_op_arguments()

    def __str__(self):
        op_code = ''
        args = self.arguments
        op_type = self.op_type
        op_code += op_name(op_type) + '('
        # add the operator arguments
        for arg in args:
            op_code += arg + ','
        # remove final comma
        op_code = op_code[:-1]
        op_code += ')'
        return op_code

    def get_op_arguments(self):
        """Extract the operator arguments from the op string and
      the previous references

  Returns
  -------
  list
      returns list of operator arguments
  """
        args = []
        op_string = self.op_string.lower()
        op_type = self.op_type
        prev_refs_code = self.prev_refs_code

        if op_type == QDMROperation.SELECT:
            # extract the entities to select
            args += [op_string]
        elif op_type == QDMROperation.FILTER:
            if extract_position_relations(op_string) != None:
                # positional relation FILTER: "#1 [relation] #2"
                relation = extract_position_relations(op_string)
                args += [relation]
                refs = extract_references(op_string)
                for ref_num in refs:
                    prev_ref_code = prev_refs_code[ref_num]
                    args += [prev_ref_code]
            else:
                # extract the reference code and the filter condition
                ref_num = extract_references(op_string)[0]
                prev_ref_code = prev_refs_code[ref_num]
                condition = op_string.split('#' + str(ref_num))[1]
                args += [prev_ref_code, condition]
        elif op_type == QDMROperation.PROJECT:
            # extract the projection attribute and the reference code
            ref_num = extract_references(op_string)[0]
            prev_ref_code = prev_refs_code[ref_num]
            projection = op_string.split('#' + str(ref_num))[0].strip()
            # remove 'of' from the projection
            if projection.endswith('of'):
                projection = projection[:-2].strip()
            args += [projection, prev_ref_code]
        elif op_type == QDMROperation.AGGREGATE:
            # extract the aggregate and the reference code
            ref_num = extract_references(op_string)[0]
            prev_ref_code = prev_refs_code[ref_num]
            aggr = extract_aggregator(op_string)
            args += [aggr, prev_ref_code]
        elif op_type == QDMROperation.GROUP:
            # extract the group operator, its values and the grouping key
            aggr = extract_aggregator(op_string)
            # split the step to the aggregated data (prefix) and key (suffix)
            data, key = op_string.split('for each')
            data_ref = extract_references(data)
            key_ref = extract_references(key)
            # check if both parts actually contained references
            if data_ref == []:
                data_code = data.split()[-1]
            else:
                data_code = prev_refs_code[data_ref[0]]
            if key_ref == []:
                key_code = key.split()[-1]
            else:
                key_code = prev_refs_code[key_ref[0]]
            args += [aggr, data_code, key_code]
        elif op_type == QDMROperation.SUPERLATIVE:
            # extract whether it is a min or max supelative
            aggr = extract_aggregator(op_string)
            entity, attribute = extract_references(op_string)
            entity_code = prev_refs_code[entity]
            attribute_code = prev_refs_code[attribute]
            args += [aggr, entity_code, attribute_code]
        elif op_type == QDMROperation.COMPARATIVE:
            # extract comparator, numeric, entity and attribute
            comparator = extract_comparator_expr(op_string)
            if comparator != 'FILTER':
                # numeric appears as the suffix of the comparative step
                comp_expr = op_string.split()[-1]
            else:
                # if FILTER comparator, return entire suffix condition
                refs = extract_references(op_string)
                condition_ref = '#' + str(refs[1])
                comp_expr = op_string.split(condition_ref)[1].strip()
            if extract_references(comp_expr) != []:
                # numeric expression is itself a reference (e.g., average of)
                refs = extract_references(comp_expr)
                assert (len(refs) == 1)
                ref = refs[0]
                comp_expr = prev_refs_code[ref]
            # extract the references
            # if contains 'where':
            if 'where #' in op_string:
                prefix, suffix = op_string.split('where')
                entity = extract_references(prefix)[0]
                attribute = extract_references(suffix)[0]
            # step does not follow 'where' satandard format
            else:
                entity, attribute = extract_references(op_string)
            entity_code = prev_refs_code[entity]
            attribute_code = prev_refs_code[attribute]
            args += [entity_code, attribute_code, comparator, comp_expr]
        elif op_type == QDMROperation.UNION:
            # add all previous references as the union arguments
            refs = extract_references(op_string)
            for ref in refs:
                args += [prev_refs_code[ref]]
        elif op_type == QDMROperation.INTERSECTION:
            interesect_expr = None
            expressions = ['of both', 'in both', 'by both']
            for expr in expressions:
                if expr in op_string:
                    interesect_expr = expr
            projection, intersection = op_string.split(interesect_expr)
            # add the projection of the intersection, e.g.,
            #  "return x in both #1 and #2"
            args += [projection]
            # add all previous references as the intersection arguments
            refs = extract_references(intersection)
            for ref in refs:
                args += [prev_refs_code[ref]]
        elif op_type == QDMROperation.DISCARD:
            # DISCARD either has the form '#x besides #y' / '#x besides something' / 'something besides #x'
            refs = extract_references(op_string)
            if len(refs) == 2:
                # '#x besides #y'
                # return the two references the source set, and the discarded set
                src, discard = extract_references(op_string)
                src_code = prev_refs_code[src]
                discard_code = prev_refs_code[discard]
            if len(refs) == 1:
                # '#x besides something' / 'something besides #x'
                prefix, suffix = op_string.split('besides')
                pref_refs = extract_references(prefix)
                if len(pref_refs) > 0:
                    # '#x besides something'
                    src = pref_refs[0]
                    src_code = prev_refs_code[src]
                    discard_code = suffix
                else:
                    # 'something besides #x'
                    src_code = prefix
                    discard = extract_references(suffix)[0]
                    discard_code = prev_refs_code[discard]
            args += [src_code, discard_code]
        elif op_type == QDMROperation.SORT:
            sorted_data_code = None
            # check type of sort expression
            sort_expr = None
            for expr in ['ordered by', 'order by', 'sorted by']:
                if expr in op_string:
                    sort_expr = expr
            assert (sort_expr != None)
            # split the sort step
            prefix, suffix = op_string.split(sort_expr)
            # extract the data to sort
            data_refs = extract_references(prefix)
            # union of data to sort
            if len(data_refs) > 1:
                refs_list = ['#' + str(ref) for ref in data_refs]
                union_string = ' or '.join(refs_list)
                sorted_data_union_code = ExecQDMR(QDMROperation.UNION, union_string, prev_refs_code)
                sorted_data_code = sorted_data_union_code
            else:
                data_ref = data_refs[0]
                sorted_data_code = prev_refs_code[data_ref]
            args += [sorted_data_code]
            # extract the sorting attribute
            sort_refs = extract_references(suffix)
            for ref in sort_refs:
                args += [prev_refs_code[ref]]
            sort_order = None
            if len(sort_refs) == 0:
                # no order reference only text 'alphabetical order'
                sort_order = suffix
            else:
                # extract the order text by removing the references
                final_ref = str(sort_refs[-1])
                sort_order = suffix.split('#' + final_ref)[1].strip()
            # add the sorting order
            args += [sort_order]
        elif op_type == QDMROperation.BOOLEAN:
            # no boolean steps in Spider dataset
            ###################!!!!!!!!!!!!!!!!!

            # "if/is/are [condition]"
            # extract the boolean condition
            condition = op_string.split()[1:]
            condition = ' '.join(condition).strip()
            # extract the condition comparator type
            refs = extract_references(condition)
            is_positional = extract_position_relations(condition)
            # check if boolean condition is numeric
            comparator_type = extract_comparator_expr(condition)
            if is_positional != None and len(refs) > 1:
                # condition regards positional relations
                condition_type = extract_position_relations(condition)
                args += [condition_type]
                for ref_num in refs:
                    arg_code = prev_refs_code[ref_num]
                    args += [arg_code]
            elif comparator_type != 'FILTER' and len(refs) <= 2:
                # numeric comparator
                first_arg = prev_refs_code[refs[0]]
                # numeric appears as the suffix of the comparative step
                comp_expr = op_string.split()[-1]
                if extract_references(comp_expr) != []:
                    # numeric expression is itself a reference (e.g., average of)
                    refs = extract_references(comp_expr)
                    assert (len(refs) == 1)
                    ref = refs[0]
                    comp_expr = prev_refs_code[ref]
                second_arg = comp_expr
                args += [comparator_type, first_arg, second_arg]
            elif ('there ' in condition) and ('any ' in condition) and \
                    (len(refs) == 1):
                # existential boolean condition - "are there any #2"
                ref = extract_references(condition)[0]
                assert (condition.endswith('#' + str(ref)))
                items = prev_refs_code[ref]
                args += ['EXIST', items]
            elif condition.endswith(' the same') and (len(refs) == 1) and \
                    (condition.startswith('all ') or condition.startswith('#') or \
                     condition.startswith('both ')):
                # distinction boolean - "are all #1 the same"
                ref = refs[0]
                items = prev_refs_code[ref]
                args += ['SAME', items]
            else:
                # FILTER condition
                return False
        elif op_type == QDMROperation.ARITHMETIC:
            arithmetics = ['sum', 'difference', 'multiplication', 'division']
            arithmetic = None
            for op in arithmetics:
                if op in op_string:
                    arithmetic = op
            assert (arithmetic != None)
            args += [arithmetic]
            # extract the arguments of the artithmetic op
            refs = extract_references(op_string)
            for ref in refs:
                args += [prev_refs_code[ref]]
            # arithmetic with constant number
            # "difference of 100 and #1"
            if len(refs) == 1:
                prefix, suffix = op_string.split('and')
                numeric_expr = prefix.split()[-1] if (extract_references(prefix) == []) else suffix.split()[0]
        elif op_type == QDMROperation.COMPARISON:
            # which is lowest of #1, #2
            comparison = extract_aggregator(op_string, True)
            assert (comparison in ['MIN', 'MAX'])
            args += [comparison]
            # extract entities to be compared
            refs = extract_references(op_string)
            for ref in refs:
                args += [prev_refs_code[ref]]
        else:
            return False
        self.arguments = args
        return self.arguments

    def to_sql(self):
        return "foo"


def extract_comparator_expr(comparative_step):
    """Extract comparator and numeric expression
     of a comparative QDMR step

 Parameters
 ----------
 comparative_step : str
     string of the QDMR comparative step

 Returns
 -------
 str
     returns string representation of the comparator expression
 """
    comparator = None
    if 'at least' in comparative_step:
        comparator = '>='
    elif 'at most' in comparative_step:
        comparator = '=<'
    elif ('more' in comparative_step) or \
            ('higher' in comparative_step) or ('larger' in comparative_step):
        comparator = '>'
    elif ('less' in comparative_step) or \
            ('smaller' in comparative_step) or ('lower' in comparative_step):
        comparator = '<'
    elif ('not ' in comparative_step) and (('same as' in comparative_step) or \
                                           ('equal' in comparative_step) or ('is' in comparative_step) or \
                                           ('was' in comparative_step) or ('are' in comparative_step)):
        comparator = '!='
    elif ('not ' not in comparative_step) and (('same as' in comparative_step) or \
                                               ('equal' in comparative_step) or ('is' in comparative_step) or \
                                               ('was' in comparative_step) or ('are' in comparative_step)) and \
            ('any' not in comparative_step):
        comparator = '='
    elif ('contain' in comparative_step):
        comparator = 'CONTAINS'
    else:
        comparator = 'FILTER'
    return comparator


def extract_aggregator(aggregate_step, include_boolean=False):
    """Extract aggregator type from QDMR aggregate step string

 Parameters
 ----------
 aggregate_step : str
     string of the QDMR aggregate step.
 include_boolean : bool
     flag whether to include true/false as operators.
     used in COMPARISON operators.

 Returns
 -------
 str
     string of the aggregate operation (sum/max/min/average/count).
 """
    if 'number of' in aggregate_step:
        return 'COUNT'
    elif ('max' in aggregate_step) or ('highest' in aggregate_step) or \
            ('largest' in aggregate_step) or ('most' in aggregate_step) or \
            ('longest' in aggregate_step) or ('biggest' in aggregate_step) or \
            ('more' in aggregate_step) or ('last' in aggregate_step) or \
            ('longer' in aggregate_step) or ('higher' in aggregate_step) or \
            ('larger' in aggregate_step):
        return 'MAX'
    elif ('min' in aggregate_step) or ('lowest' in aggregate_step) or \
            ('smallest' in aggregate_step) or ('least' in aggregate_step) or \
            ('shortest' in aggregate_step) or ('less' in aggregate_step) or \
            ('first' in aggregate_step) or ('shorter' in aggregate_step) or \
            ('lower' in aggregate_step) or ('fewer' in aggregate_step) or \
            ('smaller' in aggregate_step):
        return 'MIN'
    elif ('sum' in aggregate_step) or ('total' in aggregate_step):
        return 'SUM'
    elif ('average' in aggregate_step) or ('avg' in aggregate_step) or \
            ('mean ' in aggregate_step):
        return 'AVG'
    if include_boolean:
        if 'true ' in aggregate_step:
            return 'TRUE'
        elif 'false ' in aggregate_step:
            return 'FALSE'
        else:
            return None
    else:
        return None
    return None


# %%



def eqdmr_program(decomposition):
    """Returns an executable QDMR program representation

 Parameters
 ----------
 decomposition : str
     string representation of a QDMR

 Returns
 -------
 str
     return string representation of the executable QDMR
 """
    count = 1
    steps = parse_decomposition(decomposition)
    prev_refs_code = {}
    for step in steps:
        op_type = step_type(step, False)
        new_op = ExecQDMR(op_type, step, prev_refs_code)
        # print(count)#############
        # print(op_name(op_type))############
        # print(step)############
        # print(new_op)############
        prev_refs_code[count] = str(new_op)
        count += 1
    return prev_refs_code[count - 1]


def pretty_eqdmr(eqdmr):
    """Returns an executable QDMR program in a compositional manner

 Parameters
 ----------
 eqdmr : str
     string representation of the executable QDMR

 Returns
 -------
 bool
     return True
 """
    tab_count = 0
    pretty_represenation = ''
    for i in range(len(eqdmr)):
        if eqdmr[i] == '(':
            tab_count += 1
            pretty_represenation += '(\n'
            pretty_represenation += '\t' * tab_count
        elif eqdmr[i] == ',':
            pretty_represenation += ',\n'
            pretty_represenation += '\t' * tab_count
        elif eqdmr[i] == ')':
            tab_count -= 1
            pretty_represenation += '\n'
            pretty_represenation += '\t' * tab_count
            pretty_represenation += ')'
        else:
            pretty_represenation += eqdmr[i]
    return pretty_represenation

#
# print(eqdmr_program("return customers ;return products #1 bought ;return #1 where #2 is  food ;return names of #3"))
# pretty_print_eqdmr(
#     eqdmr_program("return customers ;return products #1 bought ;return #1 where #2 is  food ;return names of #3"))

# %%


def dataset_to_programs(dataset_path:str):
    """
    Converts dataset file to programs
    :param dataset_path: a dataset file path, e.g 'data/data_old_version/break_low_level.csv'
    :return: creates files on the same directory with the converted programs
    """
    # ERROR ANALYSIS OF SQL QDMR-TO-PROGRAM

    df = pd.read_csv('{}.csv'.format(dataset_path))
    # df = pd.read_csv('decompositions_qdmr_all.csv')
    ##valid_df = df[(df['correct']!=0)]
    decompositions = df['decomposition']

    dec_col = []
    qid_col = []
    qtext_col = []
    eqdmr_col = []

    count = 1
    for i in range(len(decompositions)):
        ###print(str(count) + '. ' + '*'*100)
        ###count += 1
        ###print(dec)
        question_id = df.loc[i, 'question_id']
        question_text = df.loc[i, 'question_text']
        dec = df.loc[i, 'decomposition']
        try:
            program = eqdmr_program(dec)
            # print(question_id)
            ###print(question_text)
            ###print(dec)
            ###print(program)
            ####pretty_print_eqdmr(program)
        except:
            dec_col += [dec]
            qid_col += [question_id]
            qtext_col += [question_text]
            eqdmr_col += ['ERROR']
        else:
            dec_col += [dec]
            qid_col += [question_id]
            qtext_col += [question_text]
            eqdmr_col += [program]
        ###print('*'*100)

    d = {'question_id': qid_col, 'question_text': qtext_col, 'decomposition': dec_col, 'program': eqdmr_col}
    programs_df = pd.DataFrame(data=d)
    programs_df.to_csv('{}__error_analysis.csv'.format(dataset_path), encoding='utf-8')
    programs_df.to_html('{}__error_analysis.html'.format(dataset_path))
    print('done...')

    # %%


def samples_to_programs(smpl_dir:str):
    """
    Add program column to predictions samples files.
    The files are assumed to be pandas.Dataframe .json/.html files, in **/*_sample directory, with "gold" and "prediction"
    columns (just like 'eval_find_interesting_samples.py' generates)
    :param smpl_dir: root directory of samples
    :return: creates a file enriched by program column for gold and prediction for each samples file in the same location
    """
    paths = [p for p in Path(smpl_dir).glob("**/*_samples/**/*.*")
             if not re.match(r".*__programs\..*$", p.name)]

    for p in paths:
        try:
            dest_dir = p.parent
            name, extension = os.path.splitext(p.name)

            if extension == ".json":
                with open(str(p), 'rt') as f:
                    df = pd.read_json(f)
            elif extension == ".html":
                df = pd.read_html(str(p))[0]
            else:
                raise Exception("Unsupported file extension {}".format(extension))

            predictions_to_programs(df, "gold")
            predictions_to_programs(df, "prediction")

            dest_path = os.path.join(dest_dir, p.name.replace(extension, "__programs{}".format(extension)))
            if extension == ".json":
                df.to_json(dest_path)
            elif extension == ".html":
                df.to_html(dest_path)
            else:
                raise Exception("Unsupported file extension {}".format(extension))

        except Exception as ex:
            print("Error on '{}'. {}".format(p, ex))
            traceback.print_exc()


def predictions_to_programs(df, qdmr_col):
    """
    Fixes the qdmr (prediction) column to a proper qdmr representation, and add a parsed program it
    :param df: pandas dataframe
    :param qdmr_col: qdmr prediction column to convert
    :return:
    """
    prog_col = "{}_program".format(qdmr_col)
    #df[qdmr_col].replace({delimiter: DELIMITER, ref:r'{}\g<1>'.format(REF)}, regex=True, inplace=True)
    df[prog_col] = df[qdmr_col]

    for index, row in df.iterrows():
        dec = row[prog_col]
        try:
            fix = prediction_to_qdmr(dec)
            df.loc[index, qdmr_col] = fix
            program = eqdmr_program(fix)
        except:
            df.loc[index, prog_col] = "ERROR"
        else:
            df.loc[index, prog_col] = program


def prediction_to_qdmr(prediction:str, add_return:bool=True):
    delimiter = "@@SEP@@"
    ref = r'@@([0-9]+)@@'
    fix = re.sub(ref, r'{}\g<1>'.format(REF), prediction)
    return DELIMITER.join(["return {}".format(d) if (not d.startswith('return ') and add_return) else d for d in fix.split(delimiter)])


def qdmr_to_prediction(qdmr:str, remove_return:bool=True):
    if not isinstance(qdmr,str):
        raise Exception("excepted string, got {}".format(str(qdmr)))
    delimiter = "@@SEP@@"
    ref = r'#([0-9]+)'
    fix = re.sub(ref, r'@@\g<1>@@', qdmr)
    return delimiter.join([re.sub(r'^\s*return\s+', '', step) if remove_return else step for step in fix.split(DELIMITER)])


def prediction_to_program(prediction: str):
    return pretty_eqdmr(eqdmr_program(prediction_to_qdmr(prediction)));



def main():
    parser = argparse.ArgumentParser(description="parse QDMR to programs in 3 available modes")
    parser.add_argument("--smpl_dir", type=str, help="root directory of samples (generated by 'eval_find_interesting_samples.py')")
    parser.add_argument("--qdmr", type=str, help="a single qdmr to parse")
    parser.add_argument("--dataset", type=str, help="parse dataset file (e.g 'data /data_old_version/break_low_level') and plots statistics")
    args = parser.parse_args()
    assert not (args.smpl_dir and args.qdmr)

    if args.qdmr:
        print(prediction_to_program(args.qdmr))
    elif args.dataset:
        dataset_to_programs(args.dataset)
    elif args.smpl_dir:
        samples_to_programs(args.smpl_dir)


if __name__ == '__main__':
    main()

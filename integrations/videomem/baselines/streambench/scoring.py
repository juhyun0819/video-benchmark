"""Deterministic label-side scoring. Never called by memory construction."""
import re


def score_answer(answer, question):
    choices=question.get('choices',[])
    expected=question.get('answer',question.get('correct_answer'))
    correct_index=question.get('correct_idx')
    if correct_index is not None:
        if type(correct_index) is not int or correct_index<0 or correct_index>=len(choices):
            raise ValueError('correct_idx must index choices')
    elif choices and expected is not None:
        matching=[i for i,value in enumerate(choices) if value.strip().casefold()==str(expected).strip().casefold()]
        if len(matching)==1:correct_index=matching[0]
        elif re.fullmatch('[A-Z]',str(expected).strip().upper()):
            index=ord(str(expected).strip().upper())-ord('A')
            if index<len(choices):correct_index=index
    if correct_index is not None:
        normalized=answer.strip().casefold()
        matching=[i for i,value in enumerate(choices) if value.strip().casefold()==normalized]
        predicted=matching[0] if len(matching)==1 else None
        match=re.fullmatch(r'\s*(?:answer\s*:\s*)?[\(\[]?([A-Z])[\)\].]?\s*',answer,re.IGNORECASE)
        if match and predicted is None:predicted=ord(match.group(1).upper())-ord('A')
        return predicted==correct_index,'multiple_choice'
    if expected is None:return None,None
    return answer.strip().casefold()==str(expected).strip().casefold(),'exact_match'

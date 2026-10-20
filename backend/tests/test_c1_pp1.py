"""Safe C1 tests for EXISTING implementation; run: python -m pytest backend/tests/test_c1_pp1.py -v"""
import sys
from pathlib import Path
import pytest

C1 = Path(__file__).resolve().parents[1] / 'app/services/c1'
sys.path.insert(0, str(C1))
from reading_metrics import (normalize_text, align_words, calculate_wpm,
                             detect_repetitions, detect_hesitations,
                             calculate_fluency_score)
from evaluation import ground_truth, labels_from_alignment, prf, evaluate_run


def test_normalize_text():
    assert normalize_text('The CAT, sat!') == ['the','cat','sat']


def test_perfect_alignment():
    r=align_words('The sheep went home.', 'the sheep went home')
    assert r['reading_accuracy'] == 100
    assert r['word_error_rate'] == 0


def test_substitution():
    r=align_words('the cat sat', 'the dog sat')
    assert r['substitution_count'] == 1


def test_omission():
    assert align_words('a b c','a c')['omission_count'] == 1


def test_insertion_and_wer():
    r=align_words('the cat sat', 'the cat cat sat')
    assert r['insertion_count'] == 1
    assert r['word_error_rate'] == pytest.approx(1/3)


def test_empty_transcription():
    assert align_words('a b c','')['omission_count'] == 3


def test_wpm():
    assert calculate_wpm(60,60) == 60
    assert calculate_wpm(60,0) is None


def test_repetition_simple():
    assert detect_repetitions('he he went')['repetition_count'] == 1


def test_no_repetition():
    assert detect_repetitions('he went home')['repetition_count'] == 0


def test_pause_excludes_boundaries():
    intervals=[{'start':0,'end':1,'duration':1},
               {'start':2,'end':2.8,'duration':0.8},
               {'start':9,'end':10,'duration':1}]
    assert detect_hesitations(intervals,10)['hesitation_count'] == 1


def test_fluency_example():
    assert calculate_fluency_score(80,70,60) == 73


def test_fluency_invalid():
    with pytest.raises(ValueError):
        calculate_fluency_score(101,70,60)


def test_prf_perfect():
    assert prf(2,0,0) == (1.,1.,1.)


def test_ground_truth_labels():
    record={'textAlignment':[
        {'promptTextWord':'a','miscueLabel':'c'},
        {'promptTextWord':'b','miscueLabel':'s'},
        {'promptTextWord':'<eps>','miscueLabel':'i'}]}
    gt=ground_truth(record)
    assert gt['n'] == 2 and gt['substitutions'] == 1 and gt['insertions'] == 1


def test_alignment_labels():
    alignment=[{'reference':'a','transcribed':'a','operation':'correct'},
               {'reference':'b','transcribed':'c','operation':'substitution'},
               {'reference':None,'transcribed':'d','operation':'insertion'}]
    assert labels_from_alignment(alignment)==['c','s']

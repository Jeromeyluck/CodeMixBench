# CodeMixBench

Code-mixing is a linguistic phenomenon where multilingual speakers switch or mix two or more languages within a single utterance or conversation. 
To evaluate LLMs’ comprehension of multilingual code-mixed texts, we introduce CodeMixBench, a benchmark comprising eight tasks across 18 languages. 

Our benchmark comprises synthesized datasets targeting knowledge reasoning, 
mathematical reasoning, and truthfulness tasks, along with LID, POS, NER, SA, and MT tasks, 
which have been adapted from open-source studies. 
For knowledge reasoning, we developed the code-mixed MMLU (CM-MMLU) based on the MMLU test set, 
featuring multiple-choice questions from 57 subjects to assess the model's comprehensive knowledge reasoning abilities. 
For mathematical reasoning, we created the code-mixed GSM8K (CM-GSM8K), derived from the GSM8K test set, 
which evaluates mathematical reasoning capabilities with each question including step-by-step solutions. 
For truthfulness assessment, we constructed the code-mixed TruthfulQA (CM-TruthfulQA) using 817 multiple-choice 
questions from the TruthfulQA test set. 

The datasets encompass 12 languages from six families: Germanic
(en, de, nl), Romance (es, fr), Sino-Tibetan (zh),
Afro-Asiatic (ar), Indo-Aryan (hi, bn, mr, ne), and
Dravidian (ta).
We synthesized 11 code-mixed language pairs for CM-
MMLU(based on the MMLU) with 12,156 question-option-answer com-
binations, 4 pairs for CM-TruthfulQA with 3,122
multiple-choice instances, 4 pairs for CM-GSM8K
with 4,367 math problems, and 3 pairs for MT with
2,711 code-mixed sentences.

## Dataset Details

### Dataset Sources

<!-- Provide the basic links for the dataset. -->

- **Paper:** CodeMixBench: Evaluating Code-Mixing Capabilities of Language Models [More Information Needed]


## Citation

<!-- If there is a paper or blog post introducing the dataset, the APA and Bibtex information for that should go in this section. -->

**BibTeX:**

[More Information Needed]

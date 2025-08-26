# Code Metric Notebook

This repo holds the metric results for both AI-generated and human-made EvalPlus implementations.
The metrics studied in this repository are:  
- __BLEU__ 
- __CodeBLEU__
- __ROUGE__
- __METEOR__
- __ChrF__

The metrics were analyzed in the order that they appear in the list above and all the observations are based on previous 
experiments; therefore, it is advisable to analyze the metrics in this exact order.

&NewLine;


## Experimental protocol

For the sake of simplicity, we measure the metrics for only one language model (*chatgpt_temp_0.8*) and only one 
HumanEval task -- *task 0*.

In order to establish the level of relevance of a given metric, we choose as a baseline the first AI-generated 
implementation that successfully passes all the tests, and then we compare it separately to all the __generated code__ 
that pass the tests and those that fail (e.g., __assert__ error, __compilation__ error, __execution__ error, __missing implementation__,
__infinite loops__, etc.)

We also consider the human-made Eval+ implementation as a baseline in order to see if the human-made and the 
AI-generated code can be differentiated using the existing metrics.

Additionally, we test the existing metrics using as a baseline an implementation for a different task, all the while 
comparing it to the **task_0** code; this kind of test should yield worse metric results considering that the semantics
of this baseline ought to be drastically different from the *task_0* implementations.


Finally, we take as a baseline another code from **chatgpt_temp_0.8 - task_0** that passes the tests and compare it to 
all the other successful implementations; considering that the __semantics__ of successful implementations for a given 
task is almost identical--with only the __syntax__ being different--the metrics should give very close results. Taking 
two successful implementations as a baseline and comparing their metric results should allow us to detect and discard 
metrics that are heavily influenced by the differences in the code's __syntax__ (e.g., variable name, instruction order,
etc.)

In order to remove any __noise__, the studied implementations are *cleaned-up* before measuring the metric: this includes 
the removal of all the __comments__ (one-line and multiple-line) and the removal of __empty lines__.

## Datasets

All the studied code and data are stored in the *__data__* folder:
- __chatgpt_temp_0.8__ holds all the AI-generated implementations of the model for the tasks 0 and 1
- __humaneval__ holds the human-made implementation Eval+ for *task_0*
- __functionality_tests__ holds the test results for the AI-generated implementations
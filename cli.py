import typer

from src.metric_test_proxy.ai_code_testing import functional_testing as ft
from src.metric_test_proxy.metric_measurement.enum import CodeDataset, TextMetric
from src.metric_test_proxy.metric_measurement import textual_metrics as tm
from src.metric_test_proxy.classifiers.enum import Classifier
from src.metric_test_proxy.classifiers.run_classifiers import run_full_classification


dataset_help = ('Dataset to test. `original` contains duplicate scripts. `distinct` is similar to `original` but with no'
                ' duplicates. Defaults to `original`.')


app = typer.Typer(help='CLI for code similarity experiments.\n\n'
                       'This project studies whether textual similarity metrics (e.g., BLEU, CodeBLEU, ROUGE) can serve '
                       'as a pre-filtering mechanism for AI-generated code: using these metrics to discard low-quality '
                       'variants and focus testing resources on the most promising scripts.')

@app.command()
def test_random_script():
    """Test a random AI-script against the according humaneval test suite."""
    ft.test_random_ai_script()


@app.command()
def test_full_dataset():
    """Run the functionality test on the entire dataset."""
    ft.full_functionality_test(CodeDataset.original)


@app.command()
def metric_score_random(
        metric: TextMetric = typer.Option(None, help='The textual metric to use. Defaults to all.'),
        functional_test: bool = typer.Option(
            False, '--run-test/--no-run-test', help='Test the AI-script.'
        )
):
    """Measure the textual similarity score on a random AI-script against the according humaneval implementation."""
    tm.random_ai_script_metrics(metric=metric, functional_test=functional_test)


@app.command()
def metric_score_full_dataset():
    """Measure the textual similarity score on the full dataset against the according humaneval implementation
     reference.
     """
    tm.full_metric_measurement(CodeDataset.original)


@app.command()
def train_test_classifier(
        dataset: CodeDataset = typer.Option(
            CodeDataset.original, help='Dataset to use. `ai_code` contains duplicate scripts. `ai_code_distinct` is '
                                       'similar but lacks any duplicates.'
        ),
        classifier: Classifier = typer.Option(None, help='The classifier to use. Defaults to all.'),
        nb_iterations: int = typer.Option(50, help='Number of iterations to train/test the model from scratch.'),
        display_results: bool = typer.Option(
            True, '--show-results/--no-show-results', help='Show classification results in the console.'
        ),
        confusion_matrix: bool = typer.Option(
            True, '--show-matrix/--no-show-matrix', help='Open the generated confusion matrix PNG.'
        ),

):
    """Train and test classification models on metrics score and functional test results of AI-scripts."""
    run_full_classification(code_dataset=dataset, classifier=classifier, nb_iterations=nb_iterations,
                            console_display=display_results, display_write_path=True, display_cm=confusion_matrix)



if __name__ == "__main__":
    app()
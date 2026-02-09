# Contributing to MRI-toolkit

Thank you for your interest in contributing to `mri-toolkit`! We welcome contributions from the community, whether it's reporting bugs, suggesting features, or submitting code changes.

This document outlines the guidelines for contributing to ensure a smooth process for everyone involved.

## Table of Contents
1. [Reporting Issues](#reporting-issues)
2. [Setting Up Your Development Environment](#setting-up-your-development-environment)
3. [Code Style and Quality](#code-style-and-quality)
4. [Running Tests](#running-tests)
5. [Submitting a Pull Request](#submitting-a-pull-request)
6. [License](#license)

## Reporting Issues

If you encounter a bug or have a feature request, please use the [GitHub Issues](https://github.com/scientificcomputing/mri-toolkit/issues) tracker.

* **Bugs:** Please provide a detailed description of the issue, including the steps to reproduce it, the expected behavior, and the actual behavior.
* **Feature Requests:** Describe the feature you would like to see and why it would be useful.

## Setting Up Your Development Environment

To start contributing, you'll need to set up a local development environment. `mri-toolkit` supports Python versions **3.10** and above.

1.  **Fork and Clone the Repository:**
    ```bash
    git clone [https://github.com/](https://github.com/)<your-username>/mri-toolkit.git
    cd mri-toolkit
    ```

2.  **Create a Virtual Environment:**
    It is recommended to use a virtual environment to manage dependencies.
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install Dependencies:**
    Install the package in editable mode along with the test dependencies.
    ```bash
    pip install -e ".[test]"
    ```

4.  **Install Pre-commit Hooks:**
    We use [pre-commit](https://pre-commit.com/) to ensure code quality before commits are made.
    ```bash
    pip install pre-commit
    pre-commit install
    ```

## Code Style and Quality

We strictly enforce code style and quality standards to maintain a clean codebase. Our CI pipeline will fail if these checks are not met.

* **Linter & Formatter:** We use **Ruff** for both linting and formatting. It is configured in `pyproject.toml` to enforce specific rules (e.g., line length of 100).
* **Type Checking:** We use **Mypy** for static type checking.
* **Pre-commit Hooks:** The following checks are run automatically on every commit via `pre-commit`:
    * Trailing whitespace removal
    * End-of-file fixing
    * YAML and TOML syntax checks
    * Ruff (linting and formatting)
    * Mypy (type checking)

To run these checks manually on all files:
```bash
pre-commit run --all-files
```
Note that these hooks will run automatically on the staged files when you commit, so you don't need to run them manually every time.


## Submitting a Pull Request
1.  **Create a New Branch:**
    ```bash
    git checkout -b feature/your-feature-name
    ```
2.  **Make Your Changes:** Implement your changes and ensure that they adhere to the code style guidelines.
3.  **Run Tests:** Make sure all tests pass before submitting your pull request.
    ```bash
    pytest
    ```
4.  **Commit Your Changes:**
    ```bash
    git add .
    git commit -m "Add a descriptive commit message"
    ```
5.  **Push Your Branch:**
    ```bash
    git push origin feature/your-feature-name
    ```
6.  **Create a Pull Request:** Go to the original repository on GitHub and create a pull request from your branch. Provide a clear description of the changes you made and any relevant information.

## Running Tests

We use **pytest** for testing. Before submitting a PR, ensure all tests pass locally.

1.  **Download Test Data:**
    Some tests require specific data. You can download this using the CLI included in the toolkit.
    ```bash
    # Downloads data to the 'test_data' folder (or your preferred location)
    python -m mritk download-test-data test_data
    ```
    *Note: You may need to set the `MRITK_TEST_DATA_FOLDER` environment variable if you download the data to a custom location.*

2.  **Run the Test Suite:**
    ```bash
    python -m pytest
    ```

    To generate a coverage report:
    ```bash
    python -m pytest --cov=mritk --cov-report html --cov-report term-missing
    ```

## Submitting a Pull Request

1.  **Create a Branch:** Create a new branch for your feature or bug fix.
    ```bash
    git checkout -b feature/my-new-feature
    ```
2.  **Make Changes:** Implement your changes. Ensure you write tests for new features or bug fixes.
3.  **Commit Changes:** Commit your changes with a descriptive message. The `pre-commit` hooks will run automatically to fix style issues.
4.  **Push:** Push your branch to your fork.
    ```bash
    git push origin feature/my-new-feature
    ```
5.  **Open a PR:** Go to the original repository and open a Pull Request.
    * The CI pipeline (GitHub Actions) will automatically run tests on Ubuntu, Windows, and macOS across Python versions 3.10–3.14.
    * Ensure all checks pass.

## License

By contributing to `mri-toolkit`, you agree that your contributions will be licensed under the [MIT License](LICENSE).

# How to Contribute to RuneLog

Thank you for your interest in contributing to Runelog! We welcome all contributions, from bug reports to new features. This guide will help you get started.

## How Can I Contribute?
- Reporting bugs: If you find a bug, please open an issue and provide a clear title, a description of the bug, steps to reproduce it, and what you expected to happen.
- Suggesting enhancements: If you have an idea for a new feature or an improvement, open an issue to start a discussion. This is the best way to ensure your idea aligns with the project's goals before you start working on it.
- Pull requests: We welcome pull requests for bug fixes, new features, and documentation improvements.

## Setting Up Your Development Environment
To get started with the code, follow these steps:

1. Fork the Repository: Click the "Fork" button on the top right of the GitHub repository page.

2. Clone Your Fork: Clone your forked repository to your local machine.

```bash
git clone https://github.com/gonz4lex/runelog.git
cd runelog
```

3. Create a Virtual Environment: We recommend using a Python virtual environment.

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`
```

4. Install in Editable Mode: Install the project and its dependencies in "editable" mode. This allows you to import your local runelog package and have any changes you make to the source code be immediately available.

```bash
pip install -e .
```

You may also need to install development dependencies like pytest if you plan to run tests.

```bash
pip install pytest
```

## Development Workflow

1. Create new branch: All contributions should be made from a feature branch. Make sure your develop branch is up-to-date, then create a new branch for your feature.

```bash
git checkout develop
git pull origin develop
git checkout -b feat/my-new-feature
```

2. Make your changes: Write your code, following the existing code style.

3. Add tests: If you are adding a new feature or fixing a bug, please consider adding tests to cover your changes. All existing tests must also pass. You can run the test suite with:

```bash
pytest
```

4. Commit the changes: RuneLog follows the [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) specification. This helps keep the project history clean and readable.

- `feat`: for a new feature.
- `fix`: for a bug fix.
- `docs`: for documentation changes.
- `refactor`: for code changes that neither fix a bug nor add a feature.
- `test`: for adding or improving tests.

5. Submit a Pull Request: Push your branch to your fork and open a pull request against the develop branch of the main repository.

### Pull Request Checklist

Before you submit your pull request, please make sure you have done the following:

- [ ] My code follows the project's style guidelines.

- [ ] I have added tests that prove my fix is effective or that my feature works.

- [ ] I have updated the documentation where necessary.

- [ ] My pull request is targeted at the develop branch.

## Code of Conduct
Please note that this project is released with a [Contributor Code of Conduct](./CODE_OF_CONDUCT.md). By participating in this project you agree to abide by its terms.

Thank you for contributing!
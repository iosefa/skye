# Contributing to Skye

Thank you for considering contributing to **Skye**, an experimental Python library for analyzing 360-degree photos in forest ecology. We greatly appreciate your support!

## Code of Conduct

We are dedicated to providing a welcoming and inclusive community. By participating, you agree to uphold this spirit and report unacceptable behavior to the maintainers.

## How to Contribute

### Reporting Issues

- Check existing issues to see if your issue has already been reported.
- If your issue is new, open a detailed report with clear steps to reproduce, expected behavior, and actual results.

### Feature Requests

- Clearly describe your idea and the problem it aims to solve.
- Include examples or mock-ups if possible.

### Submitting Pull Requests

1. Fork and clone the repository.

    ```bash
    git clone https://github.com/yourusername/skye.git
    cd skye
    ```

2. Create a new branch from `main` for your feature or bugfix:

    ```bash
    git checkout -b feature/your-feature-name
    ```

3. Install development dependencies:

    ```bash
    pip install -r requirements.txt
    ```

4. Write your code and tests.

5. Ensure tests pass and your code meets formatting guidelines (we recommend using Black):

    ```bash
    pip install black
    black .
    ```

6. Commit clearly:

    ```bash
    git commit -m "Brief description of your changes"
    ```

7. Push your branch and open a Pull Request:

    ```bash
    git push origin feature/your-feature-name
    ```

### Testing

All code should be accompanied by relevant tests. Tests should be written using Pytest.

```bash
pytest
```

## Documentation

Update or add clear documentation for any new features or changes. We use Markdown in our README and Jupyter notebooks for tutorials and examples.

## License

By contributing, you agree your contributions will be licensed under the project's MIT License.


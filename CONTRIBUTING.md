# Contributing to FeynRL

Thank you for your interest in contributing to FeynRL! We welcome contributions from the community to help make this framework more robust, efficient, and accessible.

## Development Setup

1. **Fork the Repository**: Create a fork of the FeynRL repository on GitHub.
2. **Clone your Fork**:
   ```bash
   git clone https://github.com/your-username/FeynRL.git
   cd FeynRL
   ```
3. **Set up the Environment**: Follow the instructions in [INSTALL.md](docs/INSTALL.md) to create the conda environment and install dependencies.
4. **Install Development Tools**:
   ```bash
   pip install pytest black ruff
   ```

## Contribution Process

1. **Check for Issues**: Look through the existing issues to see if what you want to work on is already being addressed.
2. **Open an Issue**: If you're proposing a major change or a new feature, please open an issue first to discuss it with the maintainers.
3. **Create a Branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```
4. **Implement your Changes**: Keep your changes focused and follow the existing code style.
5. **Add Tests**: If you're adding a new feature or fixing a bug, please include relevant tests.
6. **Run Tests**:
   ```bash
   pytest tests/
   ```
7. **Submit a Pull Request**: Once your changes are ready and tests pass, submit a PR to the `main` branch. Provide a clear description of your changes and link to any relevant issues.

## Code Style

- We use **Black** for code formatting and **Ruff** for linting.
- Follow PEP 8 guidelines.
- Use clear and descriptive variable and function names.
- Add docstrings to classes and complex functions.

## Adding New Algorithms

If you're adding a new RL or post-training algorithm:
1. Create a new directory under `algs/` (e.g., `algs/NEW_ALG/`).
2. Implement the training logic in a way that is compatible with the `Engine` and `Worker` abstractions.
3. Keep the algorithm code separate from system-level orchestration logic.
4. Add a `README.md` in your algorithm directory explaining the method and its implementation details.

## Questions?

If you have any questions about the contribution process, feel free to open an issue or reach out to the maintainers.

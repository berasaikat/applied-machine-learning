# Containerization and Continuous Integration

This repository contains the code and resources for the Containerization and Continuous Integration assignment in the Applied Machine Learning course.

## Assignment Tasks

1. **Containerization**

   - Create a Docker container for the Flask app developed in the previous assignment.
   - Create a `Dockerfile` with instructions to build the container, including:
     - Installing dependencies
     - Copying `app.py` and `score.py`
     - Launching the app by running `python app.py` upon container entry
   - Build the Docker image using the `Dockerfile`.
   - Run the Docker container with appropriate port bindings.
   - In `test_code.py`, write a `test_docker()` function that:
     - Launches the Docker container using the command line (`os.sys`, `docker build`, and `docker run`).
     - Sends a request to the localhost endpoint `/score` (e.g., using the `requests` library).
     - Checks if the response is as expected for a sample text.
     - Closes the Docker container.
   - Generate a coverage report using `pytest` for the tests in `test_code.py` and save it in `coverage.txt`.

2. **Continuous Integration**

   - Write a pre-commit Git hook that automatically runs `test_code.py` every time you try to commit code to your local `main` branch.
   - Copy and push the pre-commit Git hook file to your Git repository.

## Repository Structure

```
project_directory/
├── src/
│   ├── app.py
│   ├── score.py
│   ├── test_code.py
│   ├── model/
│      └── lightgbm_model.pkl
│   ├── requirements.txt
│   ├── Dockerfile
│   └── coverage.txt
├── .git/
│   └── hooks/
│       └── pre-commit
├── README.md
└── pre-commit.txt
```

## Dependencies

- Python
- Flask
- joblib
- pandas
- scikit-learn
- Docker

## Usage

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/your-repo.git
   ```

2. Navigate to the project directory:

   ```bash
   cd your-repo
   ```

3. Build the Docker image:

   ```bash
   docker build -t flask-app .
   ```

4. Run the Docker container:

   ```bash
   docker run -p 5000:5000 flask-app
   ```

5. Test the Flask app by sending a POST request to `http://localhost:5000/score` with a JSON payload containing the `text` field.

6. Run the tests in `test_code.py`:

   ```bash
   python3.10 -m coverage run -a -m pytest -s test_code.py
   ```

7. Generate the coverage report:

   ```bash
   python3.10 -m coverage run -m pytest
   python3.10 -m coverage report -m > coverage.txt
   ```

8. The pre-commit Git hook is located in `.git/hooks/pre-commit`. It will automatically run the tests before committing to the `main` branch.

## Contributing

Contributions are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.

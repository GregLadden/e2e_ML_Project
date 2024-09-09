# Prerequisites for Setting Up a Python Environment

## 1. **Install Python**

**macOS:**

- Install Python via [Homebrew](https://brew.sh):
  ```bash
  brew install python
  ```

**Windows:**

- Download and install Python from the [official Python website](https://www.python.org/downloads/).

**Linux:**

- Install Python using your distribution’s package manager:
  ```bash
  sudo apt-get install python3 python3-pip   # For Debian/Ubuntu
  sudo dnf install python3 python3-pip       # For Fedora
  ```

## 2. **Install Virtual Environment Tool**

**Install `venv` (built-in for Python 3.3+):**

- No separate installation is required if you’re using Python 3.3+.

**Install `virtualenv` (if using Python 2 or needing additional features):**

```bash
pip install virtualenv
```

## 3. **Create a Virtual Environment**

**Using `venv`:**

- Create a new virtual environment:
  ```bash
  python3 -m venv ml_housing_corp
  ```

## 4. **Activate the Virtual Environment**

**macOS/Linux:**

```bash
source myenv/bin/activate
```

**Windows:**

```bash
myenv\Scripts\activate

```

## 5. **Install required packages**

```
pip install -r requirements.txt
```

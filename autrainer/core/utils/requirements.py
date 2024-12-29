import os


def save_requirements(output_directory: str) -> None:
    reqs_output = os.path.join(output_directory, "requirements.txt")
    with open(reqs_output, "w") as f:
        f.write(os.popen("pip freeze").read())

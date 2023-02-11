PYTHON=python3

.PHONY: help clean tar format results report view_report static

help:	      ## Show this help message
	@sed -ne '/@sed/!s/## //p' $(MAKEFILE_LIST)

clean:	      ## Clean the directory
	git clean -dxf -e venv/ -e mlruns/ -e .idea/ -e .vscode/

tar:	      ## Tar the current project
	tar --exclude="./.git" --exclude="./__pycache" -czvf project.tar .

format:	      ## Format python files in place with black formatter
	black -l 120 .

results:      ## See Neural Net results via MLflow
	mlflow server

report:       ## Compile the report
	cd reports/project_proposal && biber report && pdflatex report.tex

view_report:  ## View PDF of report
	open reports/project_proposal/report.pdf

static:       ## Lint
	pylint --disable=C0103,C0301,R1711,R1705,R0903,R1734,W1514,C0411,R0913,R0902,R0914,R1735 .


PYTHON ?= python3

.PHONY: test check demo plan clean

test:                 ## run the offline suite (no keys, no network)
	$(PYTHON) -m unittest discover -s tests -t . -v

check:                ## validate config.toml and report which keys are present
	$(PYTHON) -m jev_cascade check

demo:                 ## end-to-end dry run: stub tiers, stub Jev, no keys needed
	$(PYTHON) -m jev_cascade demo

plan:                 ## show a decomposition without executing anything
	$(PYTHON) -m jev_cascade --config config.example.toml plan "$(TASK)" --dry-run --planner heuristic

browse:               ## report one browser step without touching the page (GOAL=... URL=...)
	$(PYTHON) -m jev_cascade browse "$(GOAL)" --url "$(URL)" --dry-run

.PHONY: test check demo plan browse clean

clean:
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
	rm -rf .pytest_cache

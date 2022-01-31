.PHONY: pyclean test test-remote test-local test-docker test-subprocess test-fast

pyclean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

test: test-local
	pytest tests ${COV_OPTIONS} -m "substra"

test-remote: pyclean
	pytest tests ${COV_OPTIONS}

test-local: pyclean
	DEBUG_SPAWNER=subprocess pytest tests ${COV_OPTIONS} --local --nbmake -m "not substra"
	DEBUG_SPAWNER=subprocess pytest tests ${COV_OPTIONS} --local -m "substra and not docker_only"
	DEBUG_SPAWNER=docker pytest tests ${COV_OPTIONS} --local -m "substra"

test-docker: pyclean
	DEBUG_SPAWNER=docker pytest tests ${COV_OPTIONS} --local

test-subprocess: pyclean
	DEBUG_SPAWNER=subprocess pytest tests ${COV_OPTIONS} --local --nbmake -m "not docker_only"

test-fast: pyclean
	DEBUG_SPAWNER=subprocess pytest tests ${COV_OPTIONS} --local --nbmake -m "not slow and not docker_only"

test-ci: pyclean
	pytest tests --ci -m "substra"

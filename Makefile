.PHONY: pyclean test test-remote test-local test-docker test-subprocess test-fast

pyclean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

# Run all the tests, all the substra tests locally in subprocess and docker mode
# and all substra tests in remote mode
test: test-local
	pytest tests ${COV_OPTIONS} -m "substra"

# Run all the tests, the substra tests are run in remote mode
test-remote: pyclean
	pytest tests ${COV_OPTIONS}

test-local: pyclean test-local-fast test-local-slow

# Run the tests, except the tests marked as slow or docker only
# The substra tests are run in local subprocess mode
test-local-fast: pyclean
	DEBUG_SPAWNER=subprocess pytest tests ${COV_OPTIONS} --local --nbmake -m "not slow and not docker_only"

# Run the slow tests in subprocess mode (those not marked docker only)
# then run all the substra tests in local docker mode
test-local-slow: pyclean
	DEBUG_SPAWNER=subprocess pytest tests ${COV_OPTIONS} --local -m "slow and not docker_only"
	DEBUG_SPAWNER=docker pytest tests ${COV_OPTIONS} --local -m "substra"

test-ci: pyclean
	pytest tests --ci -m "e2e"

benchmark: pyclean
	python benchmark/camelyon/benchmarks.py \
		--mode remote \
		--credentials-path ci.yaml \
		--nb-train-data-samples 5 \
		--nb-test-data-samples 2 \
		--batch-size 8 \
		--n-local-steps 10 \
		--n-rounds 7

benchmark-local: pyclean
	python benchmark/camelyon/benchmarks.py \
		--mode subprocess \
		--nb-train-data-samples 2 \
		--nb-test-data-samples 2 \
		--batch-size 8 \
		--n-local-steps 1 \
		--n-rounds 2

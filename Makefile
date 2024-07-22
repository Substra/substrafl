.PHONY: pyclean test test-remote test-local test-docker test-subprocess test-fast

pyclean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

# Run all the tests, all the substra tests locally in subprocess and docker mode
# and all substra tests in remote mode
test: test-local
	pytest tests ${COV_OPTIONS} -m "substra and not gpu"

# Run all the tests, the substra tests are run in remote mode
test-remote: pyclean
	pytest tests ${COV_OPTIONS} -m "not gpu and not subprocess_only"

test-local: pyclean test-subprocess-fast test-local-slow

# Run the tests, except the tests marked as slow or docker only
# The substra tests are run in local subprocess mode
test-subprocess-fast: pyclean
	pytest tests ${COV_OPTIONS} --mode=subprocess --nbmake -m "not slow and not docker_only and not gpu"

test-subprocess-slow: pyclean
	pytest tests ${COV_OPTIONS} --mode=subprocess -m "slow and not docker_only and not gpu"

test-subprocess: pyclean
	pytest tests ${COV_OPTIONS} --mode=subprocess --nbmake -m "not docker_only and not gpu"

# Run the slow tests in subprocess mode (those not marked docker only)
# then run all the substra tests in local docker mode
test-local-slow: pyclean test-subprocess-slow
	pytest tests ${COV_OPTIONS} ${PRUNE_OPTIONS} --mode=docker -m "slow and not gpu and not subprocess_only"

test-ci: pyclean
	pytest tests --ci -m "e2e and not gpu"

benchmark: pyclean
	python benchmark/camelyon/benchmarks.py \
		--mode remote \
		--credentials-path ci.yaml \
		--nb-train-data-samples 5 \
		--nb-test-data-samples 2 \
		--batch-size 4 \
		--n-local-steps 7 \
		--n-rounds 6

benchmark-local: pyclean
	python benchmark/camelyon/benchmarks.py \
		--mode subprocess \
		--nb-train-data-samples 2 \
		--nb-test-data-samples 2 \
		--batch-size 4 \
		--n-local-steps 1 \
		--n-rounds 2

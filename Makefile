.PHONY:
train:
	uv run python -m committee_predictor.train \
		--disjoint-train-ratio 0.4 \
		--neg-sampling-train-ratio 3 \
		--neg-sampling-val-test-ratio 3 \
		--thesis-filter -1000 \
		--num-epochs 300 \
		--learning-rate 0.0003 \
		--node-embedding-channels 128 \
		--hidden-channels 64 \
		--gnn-num-layers 2

.PHONY:
train_best:
	uv run python -m committee_predictor.train \
		--disjoint-train-ratio 0.4 \
		--neg-sampling-train-ratio 3 \
		--neg-sampling-val-test-ratio 3 \
		--thesis-filter -1000 \
		--num-epochs 125 \
		--learning-rate 0.0003 \
		--node-embedding-dim 128 \
		--gnn-dim 64 \
		--gnn-num-layers 2 \
		--classifier-dim 32

.PHONY:
tensorboard:
	uv run tensorboard --logdir runs

.PHONY:
install_deps_cpu:
	uv sync
	uv pip install pyg_lib -f https://data.pyg.org/whl/torch-2.8.0+cpu.html

.PHONY:
install_deps_cu128:
	uv sync
	uv pip install pyg_lib -f https://data.pyg.org/whl/torch-2.8.0+cu128.html
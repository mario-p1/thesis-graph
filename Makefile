.PHONY:
train:
	uv run python -m committee_predictor.train \
		--disjoint-train-ratio 0.6 \
		--neg-sampling-train-ratio 3 \
		--neg-sampling-val-test-ratio 3 \
		--thesis-filter -1000 \
		--num-epochs 200 \
		--learning-rate 0.0003 \
		--node-embedding-dim 128 \
		--gnn-dim 64 \
		--gnn-num-layers 2 \
		--classifier-dim 32 \
		--threshold 0.31 \
		--train-ratio 0.8 \
		--val-ratio 0.1

.PHONY:
train_best:
	uv run python -m committee_predictor.train \
		--disjoint-train-ratio 0.6 \
		--neg-sampling-train-ratio 3 \
		--neg-sampling-val-test-ratio 3 \
		--thesis-filter -1000 \
		--num-epochs 200 \
		--learning-rate 0.0003 \
		--node-embedding-dim 128 \
		--gnn-dim 64 \
		--gnn-num-layers 2 \
		--classifier-dim 32 \
		--threshold 0.31 \
		--train-ratio 0.8 \
		--val-ratio 0.1

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
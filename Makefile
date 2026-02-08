.PHONY:
train:
	uv run python -m committee_predictor.train \
		--disjoint-train-ratio 0.4 \
		--neg-sampling-train-ratio 2 \
		--neg-sampling-val-test-ratio 2 \
		--thesis-filter -1000 \
		--num-epochs 300 \
		--learning-rate 0.0003 \
		--node-embedding-channels 64 \
		--hidden-channels 64 \
		--gnn-num-layers 2


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
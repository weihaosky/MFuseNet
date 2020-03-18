
# ========== Train with 3-view multiscopic mode ============
CUDA_VISIBLE_DEVICES=0,1 python fuse.py  \
                                --trainpath='dataset/TRAIN/'    \
                                --testpath='dataset/TEST/'    \
                                --evalpath='dataset/EVAL/'    \
                                --learning_rate=1e-4 \
                                --batch_size=4  \
                                --epochs=1000   \
                                --maxdisp=60    \

# ========== Train with 5-view multiscopic mode ============
# CUDA_VISIBLE_DEVICES=0,1 python fuse_4cost.py

# ========== Train with 2-view stereo mode ==============
# CUDA_VISIBLE_DEVICES=0,1 python fuse_1cost.py
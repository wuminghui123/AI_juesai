export install_path=/usr/local/Ascend # 软件包安装路径，请根据实际修改
# driver包依赖
export RANK_ID=0
export LD_LIBRARY_PATH=/usr/local/Ascend/add-ons:$LD_LIBRARY_PATH
# fwkacllib包依赖
export LD_LIBRARY_PATH=${install_path}/fwkacllib/lib64:$LD_LIBRARY_PATH
export PYTHONPATH=${install_path}/fwkacllib/python/site-packages:${install_path}/fwkacllib/python/site-packages/auto_tune.egg/auto_tune:${install_path}/fwkacllib/python/site-packages/schedule_search.egg:$PYTHONPATH
export PATH=${install_path}/fwkacllib/ccec_compiler/bin:${install_path}/fwkacllib/bin:$PATH
# tfplugin包依赖
export PYTHONPATH=${install_path}/tfplugin/python/site-packages:$PYTHONPATH
# opp包依赖
export ASCEND_OPP_PATH=${install_path}/opp

export JOB_ID=10087
export ASCEND_DEVICE_ID=0

python eval.py
echo "See /tmp/nohup.out for jupyter url."
nohup jupyter notebook --allow-root --ip=0.0.0.0 --no-browser > /tmp/nohup.out 2>&1 &

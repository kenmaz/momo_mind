echo "dir $1"

for file in `\find $1 -maxdepth 1 -type f`; do
  cmd="python detect.py $file"
  $cmd
done

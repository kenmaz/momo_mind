echo "dir $1"

# sh run.sh ../crawler/img/ out_shiori

for file in `\find $1 -maxdepth 1 -type f`; do
  cmd="python detect.py $file $2"
  $cmd
done

allargs=( $@ )
len=${#allargs[@]}
allargs_but_first=${@:2}
argsstring=$1
nextarg=$2
nextnextarg=$3
skip_first_and_last=${allargs[@]:1:$len-2}

echo ${allargs[@]}
echo $allargs_but_first
echo $argsstring
echo $nextarg
echo $nextnextarg
echo $skip_first_and_last
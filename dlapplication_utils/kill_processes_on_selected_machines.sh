for i in "$@"
do	
	ssh dpl"$i".kdlan.iais.fraunhofer.de pkill -f "jsicking/vwplatform"
done

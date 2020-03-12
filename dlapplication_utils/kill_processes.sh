for i in 1 2 3 4 5 6
do	
	ssh dpl0"$i".kdlan.iais.fraunhofer.de pkill -f "jsicking/vwplatform"
done

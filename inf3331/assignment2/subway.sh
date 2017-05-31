#!/bin/bash

#SUBWAY BASH SCRIPT
#PLEASE READ
#-------------------------------------------------------------------------------------

#Accepted Arguments:
# ./subway.sh StationID trackID

#Example StationIDs XXXXXXX:
#Forskningsparken: 3010370
#Blindern: 3010360
#Bekkestua: 2190150


#TrackID: An integer in 0,1,....,9 (but usually 1 or 2)

#Example Usage: 
#./subway.sh 3010370 1

#If no arguments are given, it defaults to Forskningsparken, displaying for both tracknumbers

#NB! norwegian characters WILL NOT display correctly, as the instructor have pointed out.

#-------------------------------------------------------------------------------------

#Handle User Input
if (( $# < 1)); then #no arguments given
	stationID=3010370
	trackID=99
else
	#SANITY CHECKING
	#is stationID an integer of the form XXXXXXX
	stationID=3010370
	if [ ${#1} -eq 7 ]; then
		case "$1" in
			''|*[!0-9999999]*)
				;;
			*)
				stationID="$1"
				;;
		esac
	fi
	trackID=99 #default for displaying both
	#Sanity checking trackID
	if(( $# < 3)); then
		if [ ${#2} -eq 1 ]; then
			case "$2" in
				''|*[!0-9]*)
					;;
				*)
					trackID="$2"
					;;
			esac
		fi
	fi
fi

#Construct URL
url_front="http://mon.ruter.no/SisMonitor/Refresh?stopid="
url_middle=$stationID
url_end="&computerid=acba4167-b79f-4f8f-98a6-55340b1cddb3\&isOnLeftSide=true&blocks=&rows=&test=&stopPoint="
url=$url_front
url+=$url_middle
url+=$url_end

#Use the non-native command curl to get webpage
content=$(curl --silent $url)

#Assuming there are always 10 rows on the website..
for i in {0..9}; do 
	#Delete everything before subway number
	content=${content#*<td class\=\"center\">}
	#Extract subway number
	subwaynumber[i]=${content:0:1}
	#Delete everything before subway
	content=${content#*<td>}
	#Extract subway name in form *</td>
	subwayname[i]=${content%%<?td>*}
	#Delete everything before subwaytime
	content=${content#*<td>}
	content=${content#*<td>}
	#Extract subway time in form *</td>
	subwaytime[i]=${content%%<?td>*}
	#Delete everthing before subway track number
	content=${content#*<td class\=\"center\">}
	#Extract subway track number
	subwaytrack[i]=${content:0:1}
	#At this point, we are ready to extract the next subways..
done

#Display Output
for i in {0..9}; do
	if [ "${subwaytrack[i]}" == "$trackID" ] || [ "$trackID" -eq "99" ]; then
		echo ------------------------------
		echo ${subwayname[i]}
		echo ------------------------------
		echo Time: ${subwaytime[i]}
		echo Line: 	${subwaynumber[i]}
		echo Track: ${subwaytrack[i]}
	fi
done

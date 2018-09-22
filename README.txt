Installation:

Make a virtualenv, and activate it

Then:
pip install -r requrements.txt
should install the required packages.

Create a log folder

mkdir /srv/glusterfs/yawli/hashnet_logs
ln -s /srv/glusterfs/yawli/hasnet_logs logs

Then, run code with:
python hashnets.py arguments....

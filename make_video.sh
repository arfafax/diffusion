rm *.pkl
rm frames/*
mkdir frames
gsutil cp $1 .
python3 make_frames.py
ffmpeg -r 10 -i frames/foo-%02d.png -c:v libx264 -vf fps=10 -vf reverse -pix_fmt yuv420p $2.mp4


import csv

video_ends = []
last_frame = 0
for num in range(1, 81):
    with open("video" + str(num) + "-tool.csv", "r") as f:
        last_frame += int(list(f)[-1].split(",")[0])
        video_ends.append(last_frame)

with open("video_ends.csv", "w") as of:
    w = csv.writer(of)
    w.writerow(video_ends)

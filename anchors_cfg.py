
IMG_HEIGHT = 416
IMG_WIDTH = 416

CLASS_NUM = 80

ANCHORS_GROUP = {
    13: [[116, 90], [156, 198], [373, 326]],
    26: [[30, 61], [62, 45], [59, 119]],
    52: [[10, 13], [16, 30], [33, 23]]
}

# ANCHORS_GROUP = {
#     13: [[116, 90], [331,182], [334,51]],
#     26: [[30, 61], [62,45], [104,46]],
#     52: [[10, 13], [51,30], [63,29]]
# }


ANCHORS_GROUP_AREA = {
    13: [x * y for x, y in ANCHORS_GROUP[13]],
    26: [x * y for x, y in ANCHORS_GROUP[26]],
    52: [x * y for x, y in ANCHORS_GROUP[52]],
}

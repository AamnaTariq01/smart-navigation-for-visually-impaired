def detect_obstacle(distance):
    if distance < 0:
        return "Invalid distance"
    elif distance < 2:
        return "Obstacle very close"
    return "Path is clear"


def detect_obstacle_duplicate(distance):
    if distance < 0:
        return "Invalid distance"
    elif distance < 2:
        return "Obstacle very close"
    return "Path is clear"

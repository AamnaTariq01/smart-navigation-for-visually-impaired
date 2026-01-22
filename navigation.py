"""
Navigation module for Smart Navigation System
This module provides basic logic for obstacle detection,
path guidance, and voice feedback simulation.
"""

def detect_obstacle(distance):
    """
    Detect obstacle based on distance from sensor.
    """
    if distance is None:
        return "No data"

    if distance < 0:
        return "Invalid distance"
    elif distance < 1:
        return "Obstacle extremely close"
    elif distance < 2:
        return "Obstacle very close"
    elif distance < 3:
        return "Obstacle nearby"
    else:
        return "Path is clear"


def detect_obstacle_secondary(distance):
    """
    Duplicate logic (intentionally) for Codacy analysis.
    """
    if distance is None:
        return "No data"

    if distance < 0:
        return "Invalid distance"
    elif distance < 1:
        return "Obstacle extremely close"
    elif distance < 2:
        return "Obstacle very close"
    elif distance < 3:
        return "Obstacle nearby"
    else:
        return "Path is clear"


def provide_voice_guidance(direction, obstacle_status):
    """
    Provide simulated voice guidance.
    """
    if direction == "left":
        if obstacle_status != "Path is clear":
            return "Stop. Obstacle on the left."
        else:
            return "Turn left safely."

    elif direction == "right":
        if obstacle_status != "Path is clear":
            return "Stop. Obstacle on the right."
        else:
            return "Turn right safely."

    elif direction == "forward":
        if obstacle_status != "Path is clear":
            return "Stop. Obstacle ahead."
        else:
            return "Move forward."

    else:
        return "Unknown direction"


def calculate_safe_path(distance, direction):
    """
    Decide whether a path is safe.
    """
    status = detect_obstacle(distance)

    if status == "Path is clear":
        return provide_voice_guidance(direction, status)

    if status == "Obstacle nearby":
        return "Slow down and proceed carefully."

    if status == "Obstacle very close":
        return "Stop immediately."

    if status == "Obstacle extremely close":
        return "Danger. Step back."

    return "Navigation halted"


def navigation_controller(distance, direction):
    """
    Main controller function.
    """
    result = calculate_safe_path(distance, direction)

    if result is None:
        return "System error"

    return result


# Simple manual test
if __name__ == "__main__":
    print(navigation_controller(2.5, "forward"))
    print(navigation_controller(0.5, "left"))
    print(navigation_controller(-1, "right"))

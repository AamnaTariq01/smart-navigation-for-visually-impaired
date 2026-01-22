"""
Navigation module for Smart Navigation System
Provides basic logic for obstacle detection,
path guidance, and voice feedback simulation.
"""

def detect_obstacle(distance):
    """Detect obstacle based on distance from sensor."""
    if distance is None:
        return "No data"
    if distance < 0:
        return "Invalid distance"
    if distance < 1:
        return "Obstacle extremely close"
    if distance < 2:
        return "Obstacle very close"
    if distance < 3:
        return "Obstacle nearby"
    return "Path is clear"


def detect_obstacle_secondary(distance):
    """Duplicate logic (intentionally) for Codacy analysis."""
    if distance is None:
        return "No data"
    if distance < 0:
        return "Invalid distance"
    if distance < 1:
        return "Obstacle extremely close"
    if distance < 2:
        return "Obstacle very close"
    if distance < 3:
        return "Obstacle nearby"
    return "Path is clear"


def provide_voice_guidance(direction, obstacle_status):
    """Provide simulated voice guidance."""
    if direction == "left":
        return (
            "Turn left safely."
            if obstacle_status == "Path is clear"
            else "Stop. Obstacle on the left."
        )

    if direction == "right":
        return (
            "Turn right safely."
            if obstacle_status == "Path is clear"
            else "Stop. Obstacle on the right."
        )

    if direction == "forward":
        return (
            "Move forward."
            if obstacle_status == "Path is clear"
            else "Stop. Obstacle ahead."
        )

    return "Unknown direction"


def calculate_safe_path(distance, direction):
    """Decide whether a path is safe."""
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
    """Main controller function."""
    result = calculate_safe_path(distance, direction)
    return result if result is not None else "System error"


# Simple manual test
if __name__ == "__main__":
    print(navigation_controller(2.5, "forward"))
    print(navigation_controller(0.5, "left"))
    print(navigation_controller(-1, "right"))

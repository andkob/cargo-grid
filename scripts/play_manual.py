import os
import time

from warehouse_env import WarehouseEnv, EnvConfig

def clear_screen():
    os.system("cls" if os.name == "nt" else "clear")

def main():
    env = WarehouseEnv(
        EnvConfig(
            width=9,
            height=9,
            num_packages=2,
            wall_fraction=0.15,
            max_steps=300,
            battery_capacity=120,
        )
    )
    env.reset(seed=42)

    keymap = {
        "w": 0,  # up
        "s": 1,  # down
        "a": 2,  # left
        "d": 3,  # right
        "p": 4,  # pickup
        "o": 5,  # drop
    }

    try:
        import msvcrt  # Windows
        get_key = lambda: msvcrt.getwch()
        is_windows = True
    except Exception:
        is_windows = False
        get_key = None

    clear_screen()
    print("Controls: w a s d move | p pickup | o drop | r reset | q quit")
    print(env.render())

    while True:
        if is_windows:
            ch = get_key().lower() # type: ignore
        else:
            ch = input("cmd: ").strip().lower()[:1] if True else ""

        if ch == "q":
            break

        if ch == "r":
            env.reset(seed=42)
            clear_screen()
            print("Controls: w a s d move | p pickup | o drop | r reset | q quit")
            print(env.render())
            continue

        if ch not in keymap:
            continue

        obs, r, done, info = env.step(keymap[ch])

        clear_screen()
        print("Controls: w a s d move | p pickup | o drop | r reset | q quit")
        print(f"event={info.get('event')} reward={r} done={done} reason={info.get('done_reason')}")
        print(env.render())

        if done:
            print("Episode ended. Press r to reset or q to quit.")

        if is_windows:
            time.sleep(0.02)

if __name__ == "__main__":
    main()

def get_yes_no(prompt="Do you want to continue? [y/N]: "):
    while True:
        try:
            choice = input(prompt).strip().upper()
            
            if choice in ("Y", "N"):
                return choice == "Y"  # True for Yes, False for No
            else:
                print("Invalid input. Please enter 'Y' or 'N'.")
        except (EOFError, KeyboardInterrupt):
            print("\nInput interrupted. Defaulting to 'No'.")
            return False

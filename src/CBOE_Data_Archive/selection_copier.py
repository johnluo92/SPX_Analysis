import pyperclip
import time
import re
import keyboard
import threading

print("=== Real-Time Selection Monitor ===")
print("Instructions:")
print("1. Run this script")
print("2. Go to TradingView and START SELECTING text with your mouse")
print("3. As you select/highlight, it auto-captures (no need to copy)")
print("4. Keep your selection active and scroll down slowly")
print("5. Press 'ESC' key to stop capturing")
print("\nSpeed: Select about 10-20 rows at a time, pause 1 second, continue")
print("\nStarting in 3 seconds...\n")

time.sleep(3)

output_file = "PCCI_INDX_CBOE.csv"
seen_lines = set()
all_data = []
running = True

# Write header
with open(output_file, 'w') as f:
    f.write("Date,Open,High,Low,Close\n")

def stop_monitoring():
    global running
    keyboard.wait('esc')
    running = False
    print("\n[ESC pressed - stopping...]")

# Start the stop listener in a separate thread
stop_thread = threading.Thread(target=stop_monitoring, daemon=True)
stop_thread.start()

last_clipboard = ""
print("Monitoring started! Select text in TradingView now...")
print("(Press ESC to stop)\n")

try:
    while running:
        time.sleep(0.3)  # Check every 300ms
        
        try:
            # Try to get current selection via clipboard
            # Save current clipboard
            old_clipboard = pyperclip.paste()
            
            # Simulate copy to grab selection
            keyboard.press_and_release('ctrl+c')
            time.sleep(0.05)
            
            current_clipboard = pyperclip.paste()
            
            # Only process if clipboard changed
            if current_clipboard != last_clipboard and current_clipboard.strip() and current_clipboard != old_clipboard:
                last_clipboard = current_clipboard
                
                # Parse the clipboard content
                lines = current_clipboard.strip().split('\n')
                new_lines = []
                
                for line in lines:
                    # Remove the timestamp brackets
                    cleaned = re.sub(r'^\[.*?\]:\s*', '', line.strip())
                    
                    # Skip header line and empty lines
                    if cleaned and "Date,Open,High,Low,Close" not in cleaned and cleaned != "":
                        # Only add if we haven't seen this line before
                        if cleaned not in seen_lines:
                            seen_lines.add(cleaned)
                            new_lines.append(cleaned)
                
                # Append new lines to file
                if new_lines:
                    with open(output_file, 'a') as f:
                        for line in new_lines:
                            f.write(line + '\n')
                            all_data.append(line)
                    
                    print(f"✓ Captured {len(new_lines)} new lines (Total: {len(all_data)})")
        
        except Exception as e:
            # Ignore errors and continue
            pass

except Exception as e:
    print(f"\nError: {e}")

print(f"\n=== Done! ===")
print(f"Total lines captured: {len(all_data)}")
print(f"Saved to: {output_file}")
if all_data:
    print("\nFirst 3 lines:")
    for line in all_data[:3]:
        print(f"  {line}")
    if len(all_data) > 3:
        print("  ...")
        print(f"\nLast line: {all_data[-1]}")
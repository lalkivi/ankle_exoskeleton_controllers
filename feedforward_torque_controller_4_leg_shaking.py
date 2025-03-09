import matplotlib; matplotlib.use("TkAgg")
import asyncio
import math as m
import moteus
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import csv
import os
import time

max_motor_torque = 0.75 # Constrained Motor Torque Limit in Nm

#feedforward torque sine wave parameters
amplitude = 0.75 # Amplitude in Nm
frequency_hz = 7.5 # Frequency in Hz
transmission_ratio = 9.0

# Logging Control
is_logging = False  # Set to True to enable logging, False to disable
log_folder = "logs"
log_filename = "ankle_feedforward_torque_controller.csv"

# Plot Control
show_plots = True  # Set to False to disable showing plots

# Ensure the folder exists
os.makedirs(log_folder, exist_ok=True)
log_filepath = os.path.join(log_folder, log_filename)

position_list = []
velocity_list = []
desired_torque_list = []
commanded_torque_list = []
time_list = []

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 9))

def desired_torque_function(elapsed_time):
    # Calculate phase based directly on frequency and step
    phase = 2 * np.pi * frequency_hz * elapsed_time
    sine_value = amplitude * np.sin(phase)
    return sine_value

async def controller():
    c = moteus.Controller()
    s = moteus.Stream(c)
    await c.set_stop()
    start_time = time.time()
    try:
        while True:
            try:
                result = await c.query()
                position = 2 * m.pi * result.values[moteus.Register.POSITION] / transmission_ratio
                velocity = 2 * m.pi * result.values[moteus.Register.VELOCITY] / transmission_ratio
                commanded_torque = result.values[moteus.Register.TORQUE]

                # Calculate elapsed time
                elapsed_time = time.time() - start_time

                # Calculate desired torque based on the current step
                fft_torque_4_leg_shaking = desired_torque_function(elapsed_time)

                # Command the torque
                await c.set_position(position=m.nan,
                                    velocity=0.0,
                                    maximum_torque=max_motor_torque,
                                    feedforward_torque=fft_torque_4_leg_shaking,
                                    kp_scale=0.0,
                                    kd_scale=0.0,
                                    query=True)

                position_list.append(np.rad2deg(position))
                velocity_list.append(np.rad2deg(velocity))
                commanded_torque_list.append(commanded_torque)
                desired_torque_list.append(fft_torque_4_leg_shaking)
                time_list.append(elapsed_time)

            except asyncio.CancelledError:
                print("Controller loop cancelled.")
                break

            except Exception as e:
                print(f"Error in controller loop: {e}")
                await asyncio.sleep(0.00025)

    finally:
        try:
            # Stop the controller safely
            await s.write_message(b'tel stop')
            await s.flush_read()
            await s.command(b'd stop')
        except Exception as e:
            print(f"Error while stopping controller: {e}")

# Simulation and plot for the desired_torque function
step_values = np.arange(500)  # Generate step values for one sine wave period

async def main():
    try:
        await controller()
    finally:
        if is_logging:
            # Save logged data to a .csv file
            with open(log_filepath, mode="w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["Position (degrees)", "Velocity (degrees/sec)", "Commanded Torque (Nm)", "Desired Torque (Nm)"])
                for p, v, t, d_t in zip(position_list, velocity_list, commanded_torque_list, desired_torque_list):
                    writer.writerow([p, v, t, d_t])
            print(f"Data logged to {log_filepath}")

        if show_plots:
            fig.suptitle('Pendulum Swing-Up Controller Data', fontsize=12)

            ax1.plot(position_list, velocity_list)
            ax1.set_xlabel('Joint Position (degrees)', fontsize=8)
            ax1.set_ylabel('Joint Velocity (degrees/sec)', fontsize=8)
            ax1.set_title('Phase Plot', fontsize=10)
            ax1.yaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
            ax1.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
            ax1.yaxis.get_offset_text().set_position((-0.01, 0))

            ax2.plot(commanded_torque_list)
            ax2.plot(desired_torque_list)
            ax2.set_xlabel('Time Steps', fontsize=8)
            ax2.set_ylabel('Torque (Nm)', fontsize=8)
            ax2.legend(['Commanded Motor Torque', 'Desired Motor Torque'])
            ax2.set_title('Torque Plot', fontsize=10)
            ax2.yaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
            ax2.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
            ax2.yaxis.get_offset_text().set_position((-0.01, 0))

            ax3.plot(position_list)
            ax3.set_xlabel('Time Steps', fontsize=8)
            ax3.set_ylabel('Joint Position (degrees)', fontsize=8)
            ax3.set_title('Joint Position Plot', fontsize=10)

            ax4.plot(velocity_list)
            ax4.set_xlabel('Time Steps', fontsize=8)
            ax4.set_ylabel('Joint Velocity (degrees/sec)', fontsize=8)
            ax4.set_title('Joint Velocity Plot', fontsize=10)
            ax4.yaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
            ax4.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
            ax4.yaxis.get_offset_text().set_position((-0.01, 0))

            plt.tight_layout(rect=[0, 0, 1, 0.98])  # Adjust layout with title spacing
            plt.show()


if __name__ == '__main__':
    asyncio.run(main())

# At the end of every script run the hard_reset_code to exit from motor braking

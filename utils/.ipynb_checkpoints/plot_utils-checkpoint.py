import matplotlib.pyplot as plt
import pandas as pd


def plot_energy_comparison(results):
    """
    Plot energy consumption comparison between different quantization modes

    Args:
        results: Results dictionary from benchmark functions
    """
    if 'generation' in results:
        # Extract generation results
        gen_data = []
        for mode in ['fp16', 'int8', 'int4']:
            if mode in results['generation'] and 'total_energy' in results['generation'][mode]:
                gen_data.append({
                    'Mode': mode.upper(),
                    'Task': 'Generation',
                    'Energy (J)': results['generation'][mode]['total_energy'],
                    'Energy per Token (J)': results['generation'][mode]['energy_per_token'],
                    'Time (s)': results['generation'][mode]['time']
                })

        if not gen_data:
            print("No generation data to plot")
            return

        # Convert to DataFrame
        gen_df = pd.DataFrame(gen_data)

        # Plot generation energy
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        colors = {'FP16': '#3498db', 'INT8': '#2ecc71', 'INT4': '#e74c3c'}
        bars = plt.bar(gen_df['Mode'], gen_df['Energy (J)'],
                        color=[colors.get(mode, '#bbbbbb') for mode in gen_df['Mode']])

        # Add values on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{height:.2f}',
                    ha='center', va='bottom', fontsize=10)

        plt.title('Total Energy Consumption - Generation Task')
        plt.ylabel('Energy (Joules)')
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        plt.subplot(1, 2, 2)
        bars = plt.bar(gen_df['Mode'], gen_df['Energy per Token (J)'],
                        color=[colors.get(mode, '#bbbbbb') for mode in gen_df['Mode']])

        # Add values on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + height*0.05,
                    f'{height:.6f}',
                    ha='center', va='bottom', fontsize=10)

        plt.title('Energy per Token - Generation Task')
        plt.ylabel('Energy per Token (Joules)')
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        plt.tight_layout()
        plt.show()

    if 'glue' in results:
        # Extract GLUE results
        glue_data = []
        for task in results['glue']:
            for mode in ['fp16', 'int8', 'int4']:
                if mode in results['glue'][task] and 'total_energy' in results['glue'][task][mode]:
                    glue_data.append({
                        'Mode': mode.upper(),
                        'Task': task.upper(),
                        'Energy (J)': results['glue'][task][mode]['total_energy'],
                        'Energy per Token (J)': results['glue'][task][mode]['energy_per_token'],
                        'GLUE Score': results['glue'][task][mode]['glue_score']
                    })

        if not glue_data:
            print("No GLUE data to plot")
            return

        # Convert to DataFrame
        glue_df = pd.DataFrame(glue_data)

        # Plot GLUE energy - simplified for clarity
        plt.figure(figsize=(12, 5))

        # Total Energy
        plt.subplot(1, 2, 1)
        colors = {'FP16': '#3498db', 'INT8': '#2ecc71', 'INT4': '#e74c3c'}
        bars = plt.bar(glue_df['Mode'], glue_df['Energy (J)'],
                        color=[colors.get(mode, '#bbbbbb') for mode in glue_df['Mode']])

        # Add values on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{height:.2f}',
                    ha='center', va='bottom', fontsize=10)

        plt.title(f'Total Energy Consumption - {glue_df["Task"].iloc[0]} Task')
        plt.ylabel('Energy (Joules)')
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        # GLUE Score
        plt.subplot(1, 2, 2)
        bars = plt.bar(glue_df['Mode'], glue_df['GLUE Score'],
                        color=[colors.get(mode, '#bbbbbb') for mode in glue_df['Mode']])

        # Add values on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + height*0.05,
                    f'{height:.4f}',
                    ha='center', va='bottom', fontsize=10)

        plt.title(f'GLUE Score - {glue_df["Task"].iloc[0]} Task')
        plt.ylabel('Score')
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        plt.tight_layout()
        plt.show()

def plot_component_energy(results, task_type='generation', quant_mode='int4'):
    """
    Plot component-level energy breakdown

    Args:
        results: Results dictionary from benchmark
        task_type: 'generation' or 'glue'
        quant_mode: Quantization mode to plot
    """
    if task_type == 'generation':
        # Generation component breakdown
        if quant_mode in results['generation'] and 'components' in results['generation'][quant_mode]:
            components = results['generation'][quant_mode]['components']

            # Remove zero values
            components = {k: v for k, v in components.items() if v > 0}

            if not components:
                print("No component energy data to plot")
                return

            # Plot pie chart
            plt.figure(figsize=(10, 6))

            # Custom colors
            colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6']

            # Calculate percentages for labels
            total = sum(components.values())
            labels = [f"{k} ({v/total*100:.1f}%)" for k, v in components.items()]

            plt.pie(components.values(), labels=labels, colors=colors,
                   autopct='%1.1f%%', startangle=90, shadow=True)
            plt.axis('equal')
            plt.title(f'Component Energy Breakdown - Generation Task ({quant_mode.upper()})')
            plt.show()

    elif task_type == 'glue':
        # GLUE component breakdown
        task_name = list(results['glue'].keys())[0]  # Get first task
        if quant_mode in results['glue'][task_name] and 'component_energy' in results['glue'][task_name][quant_mode]:
            components = results['glue'][task_name][quant_mode]['component_energy']

            # Remove zero values
            components = {k: v for k, v in components.items() if v > 0}

            if not components:
                print("No component energy data to plot")
                return

            # Plot pie chart
            plt.figure(figsize=(10, 6))

            # Custom colors
            colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6']

            # Calculate percentages for labels
            total = sum(components.values())
            labels = [f"{k} ({v/total*100:.1f}%)" for k, v in components.items()]

            plt.pie(components.values(), labels=labels, colors=colors,
                   autopct='%1.1f%%', startangle=90, shadow=True)
            plt.axis('equal')
            plt.title(f'Component Energy Breakdown - {task_name.upper()} Task ({quant_mode.upper()})')
            plt.show()
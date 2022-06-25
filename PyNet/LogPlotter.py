import re
import matplotlib.pyplot as plt


class LogPlotter:

    @staticmethod
    def PlotLog(pattern: re.Pattern, fileName: str, title: str, yLabel: str):

        x = []
        y = []
        with open("PyNet_Logs.txt", "r") as logs:
            for line in logs:
                result = re.search(pattern, line)

                if result is not None:
                    x.append(float(result.group(1)))
                    y.append(float(result.group(2)))

        fig, ax = plt.subplots()
        ax.plot(x, y, 'o')
        ax.set_xlabel('Example Number')
        ax.set_ylabel(yLabel)
        ax.set_title(title)
        plt.savefig(f'Images/{fileName}')
        plt.close(fig)

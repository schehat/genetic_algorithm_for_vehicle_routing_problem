import os


class SavePlots:

    @staticmethod
    def save(plt, location: str, file_name: str):
        os.makedirs(location, exist_ok=True)
        file_name = os.path.join(location, file_name)
        plt.savefig(file_name)

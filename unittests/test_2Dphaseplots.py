
# Generated by Qodo Gen

# Dependencies:
# pip install pytest-mock
import pytest

class TestTwoDPhasePlot:

    # Function correctly creates a scatter plot with provided data
    def test_creates_scatter_plot_with_data(self, mocker):
        # Arrange
        from AdvancedModeling import two_D_phase_plot
        mock_plt = mocker.patch('AdvancedModeling.plt')
        test_data = [[1, 2, 3], [4, 5, 6]]
        plot_name = "test_plot"
    
        # Act
        two_D_phase_plot(test_data, plot_name)
    
        # Assert
        mock_plt.scatter.assert_called_once_with(x=test_data[0], y=test_data[1], s=1)
        mock_plt.xlabel.assert_called_once_with("N")
        mock_plt.ylabel.assert_called_once_with("N + 1")
        mock_plt.title.assert_called_once_with("Phase Plot test_plot")
        mock_plt.savefig.assert_called_once_with("2Dphase_plot_test_plot.tiff")
        mock_plt.clf.assert_called_once()
        mock_plt.close.assert_called_once()

    # Empty data lists provided as input
    def test_handles_empty_data_lists(self, mocker):
        # Arrange
        from AdvancedModeling import two_D_phase_plot
        mock_plt = mocker.patch('AdvancedModeling.plt')
        empty_data = [[], []]
        plot_name = "empty_plot"
    
        # Act
        two_D_phase_plot(empty_data, plot_name)
    
        # Assert
        mock_plt.scatter.assert_called_once_with(x=[], y=[], s=1)
        mock_plt.savefig.assert_called_once_with("2Dphase_plot_empty_plot.tiff")
        mock_plt.clf.assert_called_once()
        mock_plt.close.assert_called_once()
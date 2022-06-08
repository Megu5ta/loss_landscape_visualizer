from django import forms
from .models import NeuralNetwork


class NNForm(forms.ModelForm):
	class Meta:
		model = NeuralNetwork
		fields = ('nn_model', 'x_test', 'y_test', 'min_value', 'max_value', 'number_of_values')


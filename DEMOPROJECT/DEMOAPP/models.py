from django.db import models


class NeuralNetwork(models.Model):
	nn_model = models.FileField(upload_to='models/')
	x_test = models.FileField(upload_to='models/')
	y_test = models.FileField(upload_to='models/')
	min_value = models.FloatField()
	max_value = models.FloatField()
	number_of_values = models.IntegerField()

	def __str__(self):
		return str(self.min_value) + ' - ' + str(self.max_value) + ' : ' + str(self.number_of_values)

	def delete(self, *args, **kwards):
		self.nn_model.delete()
		self.x_test.delete()
		self.y_test.delete()
		super().delete(*args, **kwards)

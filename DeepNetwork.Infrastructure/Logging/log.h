#pragma once

#include "../matrix.h"
#include <stdio.h>
#include <stdlib.h>

void delete_log_file();
void log_matrix(matrix matrix);
void log_message(const char* message...);
void log_number(float number);
void log_whitespace();
void log_newline();
void log_float_array(float* array, int length);
void log_line(const char* message...);
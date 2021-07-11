#pragma once

#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include "log.h"

void delete_log_file() {

	FILE* fp;

	if (fp = fopen("C:\\Users\\Shaun Mitchell\\Documents\\deep_network_logs.txt", "r")) {
		fclose(fp);
		remove("C:\\Users\\Shaun Mitchell\\Documents\\matrix_multiple.txt");
	}
}

void log_matrix(matrix matrix) {
	FILE* fp = fopen("C:\\Users\\Shaun Mitchell\\Documents\\deep_network_logs.txt", "a");

	for (int i = 0; i < matrix.rows; i++) {
		for (int j = 0; j < matrix.cols; j++) {
			char buffer[100];
			gcvt(matrix.values[matrix.cols * i + j], 20, buffer);
			fwrite(buffer, sizeof(char), 20, fp);

			if (j != matrix.cols - 1) {
				fwrite(", ", sizeof(char), 2, fp);
			}
		}

		fwrite("\n", sizeof(char), 1, fp);
	}

	fwrite("\n", sizeof(char), 1, fp);
	fwrite("\n", sizeof(char), 1, fp);
	fwrite("\n", sizeof(char), 1, fp);
	fwrite("\n", sizeof(char), 1, fp);

	fclose(fp);
}

static void log_message_without_date(const char* message...) {
	FILE* fp = fopen("C:\\Users\\Shaun Mitchell\\Documents\\deep_network_logs.txt", "a");
	fprintf(fp, message);
	fclose(fp);
}

void log_message_classic(const char* message) {
	FILE* fp = fopen("C:\\Users\\Shaun Mitchell\\Documents\\deep_network_logs.txt", "a");

	fwrite(message, sizeof(char), 1, fp);
	fwrite("\n", sizeof(char), 1, fp);
	fclose(fp);
}

void log_message(const char* message...) {
	FILE* fp = fopen("C:\\Users\\Shaun Mitchell\\Documents\\deep_network_logs.txt", "a");
	
	time_t t = time(NULL);
	struct tm tm = *localtime(&t);
	fprintf(fp, "%d-%02d-%02d %02d:%02d:%02d | ", tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday, tm.tm_hour, tm.tm_min, tm.tm_sec);
	fprintf(fp, message);
	fclose(fp);
}



void log_number(float number) {
	char buffer[sizeof(float)];
	log_message_classic(gcvt(number, sizeof(float), buffer));
}

void log_whitespace() {
	log_message_without_date(" ");
}

void log_newline() {
	log_message_without_date("\n");
}

void log_float_array(float* array, int length) {
	for (int i = 0; i < length; i++) {
		log_number(array[i]);
		log_whitespace();
	}
	log_newline();
}

void log_line(const char* message...) {
	log_message(message);
	log_newline();
}
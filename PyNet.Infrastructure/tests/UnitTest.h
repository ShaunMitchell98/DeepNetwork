#pragma once

#include <Setup.h>
#include "Settings.h"
#include <gtest/gtest.h>
#include <memory>

namespace PyNet::Infrastructure::Tests {

	class UnitTest : public ::testing::Test
	{
	private:
		shared_ptr<Context> _context;

	protected:

		void SetUp() override {
			auto settings = make_shared<Settings>();
			settings->LoggingEnabled = false;
			_context = GetContext(settings, false);
		}

		template<class T>
		std::unique_ptr<T> GetUniqueService() {
			return _context->GetUnique<T>();
		}

		template<class T>
		std::shared_ptr<T> GetSharedService() {
			return _context->GetShared<T>();
		}
	};
}
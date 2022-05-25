#pragma once

#include "Settings.h"
#include "ContainerFixture.h"
#include <gtest/gtest.h>
#include <memory>

using namespace std;

namespace PyNet::Infrastructure::Tests {

	class UnitTest : public ::testing::Test
	{
	private:
		shared_ptr<Context> _context;

	protected:

		void SetUp() override 
		{
			_context = ContainerFixture::Initialise();
		}

		template<class T>
		unique_ptr<T> GetUniqueService() 
		{
			return _context->GetUnique<T>();
		}

		template<class T>
		shared_ptr<T> GetSharedService()
		{
			return _context->GetShared<T>();
		}
	};
}
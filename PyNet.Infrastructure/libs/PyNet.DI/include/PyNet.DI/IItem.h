#pragma once
 #include <memory>
 #include <any>

 using namespace std;

 class IItem {
 public:
 	virtual int GetUsageCount() = 0;
 	virtual void Reset() = 0;
	virtual void* GetInstance(any& context) = 0;
	virtual bool HasInstance() = 0;
    virtual void MakeReferenceWeak() = 0;
	bool Marker;
 };